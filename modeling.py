import logging
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import BertForTokenClassification
from models.centroid_loss import cos_centroidsLoss

from models.MI_estimators import VIB, vCLUB, InfoNCE

logger = logging.getLogger(__file__)


class BertForTokenClassification_(BertForTokenClassification):
    def __init__(self, *args, **kwargs):
        super(BertForTokenClassification_, self).__init__(*args, **kwargs)
        self.input_size = 768
        self.span_loss = nn.functional.cross_entropy
        self.type_loss = nn.functional.cross_entropy
        self.dropout = nn.Dropout(p=0.1)
        self.log_softmax = nn.functional.log_softmax

        self.f_centroids = None

        self.oov_reg = None
        self.z_reg = None

    def set_config(
            self,
            use_classify: bool = False,
            distance_mode: str = "cos",
            similar_k: float = 30,
            shared_bert: bool = True,
            train_mode: str = "add",
            n_type=0,
            device=None
    ):

        if train_mode == 'span':
            n_base_class = 5
            out_dim = 768
            n_centroids = 15
            device = 'cuda'
            T = 0.025
            m = 0.01
            self.f_centroids = cos_centroidsLoss(n_base_class, out_dim, n_centroids, device, T, m)
        elif train_mode == 'type':

            self.gama = 0.001
            self.z_reg = InfoNCE(768, 768, device)
            self.beta = 0.00001
            self.oov_reg = vCLUB()

        elif train_mode == 'add':
            pass
        else:
            raise ValueError("Invalid mode!")

        self.use_classify = use_classify
        self.distance_mode = distance_mode
        self.similar_k = similar_k
        self.shared_bert = shared_bert
        self.train_mode = train_mode
        if train_mode == "type":
            self.classifier = None

        if train_mode != "span":
            self.ln = nn.LayerNorm(768, 1e-5, True)
            if use_classify:
                self.type_classify = nn.Sequential(
                    nn.Linear(self.input_size, self.input_size * 2),
                    nn.GELU(),
                    nn.Linear(self.input_size * 2, self.input_size),
                )
            if self.distance_mode != "cos":
                self.dis_cls = nn.Sequential(
                    nn.Linear(self.input_size * 3, self.input_size),
                    nn.GELU(),
                    nn.Linear(self.input_size, 2),
                )
        config = {
            "use_classify": use_classify,
            "distance_mode": distance_mode,
            "similar_k": similar_k,
            "shared_bert": shared_bert,
            "train_mode": train_mode,
        }
        logger.info(f"Model Setting: {config}")
        if not shared_bert:
            self.bert2 = deepcopy(self.bert)

    def forward_wuqh(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            e_mask=None,
            e_type_ids=None,
            e_type_mask=None,
            entity_types=None,
            entity_mode: str = "mean",
            is_update_type_embedding: bool = False,
            lambda_max_loss: float = 0.0,
            sim_k: float = 0,
            span_all=None,
            span_all_norm=None
    ):
        max_len = (attention_mask != 0).max(0)[0].nonzero(as_tuple=False)[-1].item() + 1
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len].type(torch.int8)
        token_type_ids = token_type_ids[:, :max_len]
        labels = labels[:, :max_len]
        output = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_output = self.dropout(output[0])

        # =====centroids=====
        if self.f_centroids != None:
            norm_z, norm_c = self.f_centroids.forward(
                sequence_output.view(sequence_output.shape[0],
                                     -1))
            norm_z = norm_z.view(sequence_output.shape[0], -1, sequence_output.shape[2])

        if self.train_mode != "type":
            logits = self.classifier(
                sequence_output
            )
        else:
            logits = None
        if not self.shared_bert:
            output2 = self.bert2(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )
            sequence_output2 = self.dropout(output2[0])

        if e_type_ids is not None and self.train_mode != "span":
            if e_type_mask.sum() != 0:
                M = (e_type_mask[:, :, 0] != 0).max(0)[0].nonzero(as_tuple=False)[
                        -1
                    ].item() + 1
            else:
                M = 1
            e_mask = e_mask[:, :M, :max_len].type(torch.int8)
            e_type_ids = e_type_ids[:, :M, :]
            e_type_mask = e_type_mask[:, :M, :].type(torch.int8)
            B, M, K = e_type_ids.shape

            e_out, feat_entity, feat_context = self.get_enity_hidden(
                sequence_output if self.shared_bert else sequence_output2,
                e_mask,
                entity_mode,
            )
            if span_all != None:
                span_rep = e_out.clone().detach().cpu().numpy().tolist()
                span_label = e_type_ids.clone().detach().cpu().numpy().tolist()
                span_all.append([span_rep, span_label])
            if self.use_classify:
                e_out = self.type_classify(e_out)
            e_out = self.ln(e_out)
            if span_all_norm != None:
                span_rep_norm = e_out.clone().detach().cpu().numpy().tolist()
                span_label = e_type_ids.clone().detach().cpu().numpy().tolist()
                span_all_norm.append([span_rep_norm, span_label])
            if is_update_type_embedding:
                entity_types.update_type_embedding(e_out, e_type_ids, e_type_mask)
            e_out = e_out.unsqueeze(2).expand(B, M, K, -1)
            types = self.get_types_embedding(
                e_type_ids, entity_types
            )

            if self.distance_mode == "cat":
                e_types = torch.cat([e_out, types, (e_out - types).abs()], -1)
                e_types = self.dis_cls(e_types)
                e_types = e_types[:, :, :0]
            elif self.distance_mode == "l2":
                e_types = -(torch.pow(e_out - types, 2)).sum(-1)
            elif self.distance_mode == "cos":
                sim_k = sim_k if sim_k else self.similar_k
                e_types = sim_k * (e_out * types).sum(-1) / 768

            e_logits = e_types
            if M:
                em = e_type_mask.clone()
                em[em.sum(-1) == 0] = 1
                e = e_types * em
                e_type_label = torch.zeros((B, M)).to(e_types.device)
                type_loss = self.calc_loss(
                    self.type_loss, e, e_type_label, e_type_mask[:, :, 0]
                )

                # =======MI(type)=======
                loss_gi = self.gama * self.z_reg(feat_entity, feat_context)
                loss_si = self.beta * (self.oov_reg.update(feat_entity, feat_context))
                type_loss += loss_gi
                type_loss += loss_si
            else:
                type_loss = torch.tensor(0).to(sequence_output.device)
        else:
            e_logits, type_loss = None, None

        if labels is not None and self.train_mode != "type":
            # Only keep active parts of the loss
            loss_fct = CrossEntropyLoss(reduction="none")
            B, M, T = logits.shape
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.reshape(-1, self.num_labels)[active_loss]
                active_labels = labels.reshape(-1)[active_loss]
                base_loss = loss_fct(active_logits, active_labels)
                loss = torch.mean(base_loss)

                # max-loss
                if lambda_max_loss > 0:
                    active_loss2 = active_loss.view(B, M)
                    active_max = []
                    start_id = 0
                    for i in range(B):
                        sent_len = torch.sum(active_loss2[i])
                        end_id = start_id + sent_len
                        active_max.append(torch.max(base_loss[start_id:end_id]))
                        start_id = end_id

                    loss += lambda_max_loss * torch.mean(torch.stack(active_max))

                    # =======centroids loss (span)=======
                active_labels2 = active_labels.clone()
                for i_lab in range(active_labels2.shape[0]):
                    if active_labels2[i_lab] < 0:
                        active_labels2[i_lab] = 0

                active_norm_z = norm_z.reshape(-1, norm_z.shape[2])[active_loss]

                yc, yn, ys = self.f_centroids.labeling(active_norm_z, norm_c, active_labels2,
                                                       self.f_centroids.n_centroids)
                proto_c = norm_c.view(-1, self.f_centroids.n_centroids, norm_c.shape[1]).mean(1)

                colloboration_loss = self.f_centroids.loss(active_norm_z, proto_c.detach(),
                                                           active_labels2)

                competition_loss = self.f_centroids.loss(active_norm_z, norm_c[yc], yn)
                loss += (0.1 * colloboration_loss)
                loss += (0.1 * competition_loss)
            else:
                raise ValueError("Miss attention mask!")
        else:
            loss = None

        return logits, e_logits, loss, type_loss


    def get_enity_hidden(
            self, hidden: torch.Tensor, e_mask: torch.Tensor, entity_mode: str
    ):
        B, M, T = e_mask.shape

        e_out = hidden.unsqueeze(1).expand(B, M, T, -1) * e_mask.unsqueeze(
            -1
        )

        e_mask_context = torch.ones_like(e_mask) - e_mask
        e_out_context = hidden.unsqueeze(1).expand(B, M, T, -1) * e_mask_context.unsqueeze(
            -1
        )

        if entity_mode == "mean":
            feat_entity = (e_out.sum(2) / (e_mask.sum(-1).unsqueeze(-1) + 1e-30))
            feat_context = (e_out_context.sum(2) / (e_mask_context.sum(-1).unsqueeze(-1) + 1e-30))

            return feat_entity, feat_entity.mean(dim=1), feat_context.mean(
                dim=1)

    def get_types_embedding(self, e_type_ids: torch.Tensor, entity_types):
        return entity_types.get_types_embedding(e_type_ids)

    def calc_loss(self, loss_fn, preds, target, mask=None):
        target = target.reshape(-1)
        preds += 1e-10
        preds = preds.reshape(-1, preds.shape[-1])
        ce_loss = loss_fn(preds, target.long(), reduction="none")
        if mask is not None:
            mask = mask.reshape(-1)
            ce_loss = ce_loss * mask
            return ce_loss.sum() / (mask.sum() + 1e-10)
        return ce_loss.sum() / (target.sum() + 1e-10)


class ViterbiDecoder(object):
    def __init__(
            self,
            id2label,
            transition_matrix,
            ignore_token_label_id=torch.nn.CrossEntropyLoss().ignore_index,
    ):
        self.id2label = id2label
        self.n_labels = len(id2label)
        self.transitions = transition_matrix
        self.ignore_token_label_id = ignore_token_label_id

    def forward(self, logprobs, attention_mask, label_ids):

        batch_size, max_seq_len, n_labels = logprobs.size()
        attention_mask = attention_mask[:, :max_seq_len]
        label_ids = label_ids[:, :max_seq_len]

        active_tokens = (attention_mask == 1) & (
                label_ids != self.ignore_token_label_id
        )
        if n_labels != self.n_labels:
            raise ValueError("Labels do not match!")

        label_seqs = []

        for idx in range(batch_size):
            logprob_i = logprobs[idx, :, :][
                active_tokens[idx]
            ]

            back_pointers = []

            forward_var = logprob_i[0]

            for j in range(1, len(logprob_i)):
                next_label_var = forward_var + self.transitions
                viterbivars_t, bptrs_t = torch.max(next_label_var, dim=1)

                logp_j = logprob_i[j]
                forward_var = viterbivars_t + logp_j
                bptrs_t = bptrs_t.cpu().numpy().tolist()
                back_pointers.append(bptrs_t)

            path_score, best_label_id = torch.max(forward_var, dim=-1)
            best_label_id = best_label_id.item()
            best_path = [best_label_id]

            for bptrs_t in reversed(back_pointers):
                best_label_id = bptrs_t[best_label_id]
                best_path.append(best_label_id)

            if len(best_path) != len(logprob_i):
                raise ValueError("Number of labels doesn't match!")

            best_path.reverse()
            label_seqs.append(best_path)

        return label_seqs
