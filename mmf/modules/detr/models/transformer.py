# Copyright (c) Facebook, Inc. and its affiliates.

# Mostly copy-pasted from
# https://github.com/facebookresearch/detr/blob/master/models/transformer.py
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Transformer(nn.Module):
    def __init__(
        self,
        args,
        d_model_enc=512,
        d_model_dec=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        pass_pos_and_query=True,
        share_decoders=False,
    ):
        super().__init__()

        self.args = args

        self.pass_pos_and_query = pass_pos_and_query
        encoder_layer = TransformerEncoderLayer(
            d_model_enc, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model_enc) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        if d_model_dec != d_model_enc:
            self.enc2dec_proj = nn.Linear(d_model_enc, d_model_dec)
            self.pos_embed_proj = nn.Linear(d_model_enc, d_model_dec)
        else:
            self.enc2dec_proj = nn.Identity()
            self.pos_embed_proj = nn.Identity()

        decoder_layer = TransformerDecoderLayer(
            d_model_dec, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model_dec)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model_enc = d_model_enc
        self.d_model_dec = d_model_dec
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, src, mask, query_embed, pos_embed, query_embed_vqa=None, task_idx=None
    ):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        if self.pass_pos_and_query:
            tgt = torch.zeros_like(query_embed)
        else:
            src, tgt, query_embed, pos_embed = (
                src + 0.1 * pos_embed,
                query_embed,
                None,
                None,
            )
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        if self.args.residual_in_encoder:
            memory = src + memory
        memory = self.enc2dec_proj(memory)
        pos_embed = self.pos_embed_proj(pos_embed)
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w), None


class MTTransformer(Transformer):
    def __init__(
        self,
        args,
        d_model_enc=512,
        d_model_dec=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        pass_pos_and_query=True,
        share_decoders=False,
    ):
        super().__init__(
            args=args,
            d_model_enc=d_model_enc,
            d_model_dec=d_model_dec,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
            return_intermediate_dec=return_intermediate_dec,
            pass_pos_and_query=pass_pos_and_query,
        )
        # text_src is already projected from nn.Linear, so we don't need to add an extra
        # segment embedding (the bias term in nn.Linear can learn this embedding)
        # self.segment_projection = nn.Linear(d_model_dec, d_model_dec)
        self.share_decoders = share_decoders
        num_queries = self.args.num_queries

        self.decoders = nn.ModuleDict()
        for task in num_queries:
            task_dict = nn.ModuleDict()
            for dataset in num_queries[task]:
                if share_decoders:
                    task_dict[dataset] = self.decoder
                else:
                    self.decoder = None
                    task_dict[dataset] = self.build_decoder_layer(
                        d_model_dec=d_model_dec,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        activation=activation,
                        normalize_before=normalize_before,
                        num_decoder_layers=num_decoder_layers,
                        return_intermediate_dec=return_intermediate_dec,
                    )
            self.decoders[task] = task_dict
            # A separate decoder for VQA

        MAX_TASK_NUM = 256
        if args.use_task_embedding_in_encoder:
            self.task_embeddings_enc = nn.Embedding(MAX_TASK_NUM, d_model_enc)
        if args.use_task_embedding_in_decoder:
            self.task_embeddings_dec = nn.Embedding(MAX_TASK_NUM, d_model_dec)
        # when adding the task embedding to the beginning of the decoder, we'll strip
        # it from the hidden state outputs to make it compatible with previous models
        self.mem_out_begin_idx = 1 if args.use_task_embedding_in_encoder else 0
        self.hs_out_begin_idx = 1 if args.use_task_embedding_in_decoder else 0

    def build_decoder_layer(
        self,
        d_model_dec=512,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        decoder_layer = TransformerDecoderLayer(
            d_model_dec, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model_dec)
        return TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

    def forward(
        self,
        img_src=None,
        img_mask=None,
        img_pos=None,
        text_src=None,
        text_mask=None,
        text_pos=None,
        query_embed=None,
        task_type=None,
        dataset_name=None,
        task_idx=None,
    ):
        # flatten NxCxHxW to HWxNxC
        memories = []
        pos_embeds = []
        masks = []

        if img_src is not None:
            bs, c, h, w = img_src.shape
            img_src = img_src.flatten(2).permute(2, 0, 1)
            img_pos = img_pos.flatten(2).permute(2, 0, 1)
            img_mask = img_mask.flatten(1)
            if text_src is None:
                query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
                query_embed = self._prefix_task_embedding_to_query_embed(
                    query_embed, task_idx
                )
                if self.pass_pos_and_query:
                    tgt = torch.zeros_like(query_embed)
                else:
                    img_src, tgt, query_embed, img_pos = (
                        img_src + 0.1 * img_pos,
                        query_embed,
                        None,
                        None,
                    )
            img_src, img_mask, img_pos = self._prefix_task_embedding_to_encoder_inputs(
                img_src, img_mask, img_pos, task_idx
            )
            memory = self.encoder(img_src, src_key_padding_mask=img_mask, pos=img_pos)

            if self.mem_out_begin_idx != 0:
                img_src = img_src[self.mem_out_begin_idx :]
                img_pos = img_pos[self.mem_out_begin_idx :]
                img_mask = img_mask[:, self.mem_out_begin_idx :]
                memory = memory[self.mem_out_begin_idx :]

            if self.args.residual_in_encoder:
                memory = img_src + memory

            memory = self.enc2dec_proj(memory)
            img_pos = self.pos_embed_proj(img_pos)
            memories.append(memory)
            pos_embeds.append(img_pos)
            masks.append(img_mask)

        if text_src is not None:
            text_src = text_src.permute(1, 0, 2)
            memories.append(text_src)
            text_pos = text_pos.unsqueeze(1).repeat(1, text_src.size(1), 1)
            pos_embeds.append(text_pos)
            masks.append(text_mask != 1)

            query_embed = query_embed.unsqueeze(1).repeat(1, text_src.size(1), 1)
            query_embed = self._prefix_task_embedding_to_query_embed(
                query_embed, task_idx
            )
            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                raise NotImplementedError()

        decoder = self.decoders[task_type][dataset_name]

        memories = torch.cat(memories)
        masks = torch.cat(masks, dim=-1)
        pos_embeds = torch.cat(pos_embeds)

        hs = decoder(
            tgt,
            memories,
            memory_key_padding_mask=masks,
            pos=pos_embeds,
            query_pos=query_embed,
        )
        hs = hs.transpose(1, 2)
        # hs is num_layer x batch_size x seq_length x hidden_dim
        hs = hs[:, :, self.hs_out_begin_idx :, :]

        return hs, memories.permute(1, 2, 0)

    def _prefix_task_embedding_to_query_embed(self, query_embed, task_idx):
        if not self.args.use_task_embedding_in_decoder:
            return query_embed

        bs = query_embed.size(1)
        task_embed = self.task_embeddings_dec.weight[task_idx]
        task_embed = task_embed.unsqueeze(0).unsqueeze(0).repeat(1, bs, 1)
        query_embed = torch.cat([task_embed, query_embed], dim=0)
        return query_embed

    def _prefix_task_embedding_to_encoder_inputs(
        self, img_src, img_mask, img_pos, task_idx
    ):
        if not self.args.use_task_embedding_in_encoder:
            return img_src, img_mask, img_pos

        bs = img_src.size(1)
        task_embed = self.task_embeddings_enc.weight[task_idx]
        task_embed = task_embed.unsqueeze(0).unsqueeze(0).repeat(1, bs, 1)
        img_src = torch.cat([task_embed, img_src], dim=0)

        # 0 for non-padding in img_mask
        img_mask_pad = torch.zeros_like(img_mask[:, :1])
        img_mask = torch.cat([img_mask_pad, img_mask], dim=1)
        img_pos_pad = torch.zeros_like(img_pos[:1])
        img_pos = torch.cat([img_pos_pad, img_pos], dim=0)

        return img_src, img_mask, img_pos


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    # TODO(ronghanghu): remove all other parameters and use args only
    return MTTransformer(
        args=args,
        d_model_enc=args.encoder_hidden_dim,
        d_model_dec=args.decoder_hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        pass_pos_and_query=args.pass_pos_and_query,
        share_decoders=args.share_decoders,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
