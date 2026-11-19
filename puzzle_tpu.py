#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
puzzle_tpu.py — portage JAX/TPU de docta.cpp / puzzle.cu (recherche de clé
privée secp256k1 par force brute, par fenêtre centrale + inversion par lots).

LIS CECI D'ABORD — ce que ce portage fait et ne fait PAS :

  TPU = mauvaise cible pour ce calcul. Un TPU est une unité de multiplication
  de matrices (bf16/fp32) ; il n'a ni "cœurs" indépendants avec branchement
  dynamique (comme un GPU), ni opérations 128 bits, ni primitives bit à bit
  rapides au sens où un CPU/GPU les a. Toute cette recherche (arithmétique
  modulaire 256 bits, SHA-256, RIPEMD-160, filtre de Bloom, blocs pris de
  façon atomique) est faite d'opérations entières/bit à bit et de branchements
  données-dépendants : rien de tout cela n'utilise la MXU (l'unité qui fait la
  puissance d'un TPU), et XLA/TPU interdit de toute façon les formes/boucles
  dynamiques qu'utilisait le code original (claim_block atomique, sortie
  anticipée par thread, ...). Un portage "ligne à ligne" est donc impossible :
  ce fichier est une RÉÉCRITURE qui adapte l'algorithme à ce que XLA accepte
  (tout est vectorisé par lot, formes statiques, pas de court-circuit).

  Concrètement, par rapport à docta.cpp :
    - gardé   : l'astuce centrale (inversion de corps par lots de 2*HALF clés
                autour d'un point central), l'arithmétique secp256k1, le
                hachage hash160, les adresses P2PKH/bech32, le puzzle -pN.
    - changé  : le parallélisme "threads CPU/GPU" devient un axe de lot (B
                "lanes" indépendantes, vectorisées par JAX/XLA — l'équivalent
                direct des threads CUDA de puzzle.cu, voir g_thread/g_kernel).
    - simplifié / retiré (pour rester dans un temps de développement
      raisonnable et dans ce que XLA accepte facilement) :
        * modes Sobol / Monte Carlo / pi / e / ln -> seulement séquentiel et
          aléatoire (le même mécanisme s'étend facilement aux autres, cf. la
          fonction `next_chunk`).
        * voisinage -v R.
        * filtre de Bloom + fichier .bloom -> comparaison directe à la liste
          des cibles (très bien pour 1 à quelques milliers de cibles, qui est
          le cas d'usage réel : le puzzle Bitcoin a UNE adresse par niveau).
        * le cas dégénéré "C = ±i·G" (probabilité ~2*HALF / ordre de la
          courbe, donc ~0 en pratique) : au lieu du recalcul exact, on
          ignore juste la clé concernée pour cette fenêtre (voir group_step).
        * la fin de plage n'est respectée qu'à 2*HALF clés près (une fenêtre
          peut légèrement dépasser la borne demandée).

  Honnêteté sur la performance : int64/uint64 ne sont pas le point fort d'un
  TPU (ils sont supportés par XLA mais souvent ÉMULÉS avec des paires
  d'opérations 32 bits), et ce code en a besoin pour les retenues de
  multiplication 32x32->64. Attends-toi à un gain bien plus modeste que sur
  GPU, voire à ce qu'un GPU (ou même des CPU nombreux) batte un TPU sur cette
  tâche précise. Ce script est un VRAI portage fonctionnel, pas une promesse
  de performance.

  AVANT une vraie recherche, lance toujours :
      python3 puzzle_tpu.py --selftest
  (ça vérifie, sur de petits cas, que la version JAX donne EXACTEMENT les
  mêmes résultats que l'arithmétique de référence en entiers Python.)

Usage (voir aussi --help) :
  python3 puzzle_tpu.py -p 40                      # puzzle Bitcoin n°40
  python3 puzzle_tpu.py <adresse|pubkey> <bits>    # ex. plage 2^39..2^40-1
  python3 puzzle_tpu.py <adresse> <début:fin> address
"""
import argparse
import os
import random
import sys
import time

import btc_host as H

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)   # indispensable : retenues de multiplication 32x32->64 bits
    HAVE_JAX = True
except Exception as e:  # pragma: no cover
    HAVE_JAX = False
    _JAX_IMPORT_ERROR = e

MASK32 = None  # défini après import jax (voir plus bas), pour que le fichier reste important-able sans jax (host-only)


# ============================================================ arithmétique Fp (JAX), limbes 32 bits
# Même algorithme que gf_add/gf_sub/gf_mul/gf_reduce/gf_inv de puzzle.cu (struct FE { u32 v[8] }),
# vectorisé sur un lot de B "lanes" indépendantes (axe 0). p = 2^256 - 2^32 - 977.

def _install_fe_ops():
    global MASK32, fe_add, fe_sub, fe_mul, fe_sqr, fe_sqr_n, fe_inv, fe_is_zero
    MASK32 = jnp.uint64(0xFFFFFFFF)

    def fe_add(a, b):
        B = a.shape[0]
        ac = [a[:, i] for i in range(8)]
        bc = [b[:, i] for i in range(8)]
        t = [None] * 8
        c = jnp.zeros((B,), dtype=jnp.uint64)
        for i in range(8):
            c = c + ac[i].astype(jnp.uint64) + bc[i].astype(jnp.uint64)
            t[i] = (c & MASK32).astype(jnp.uint32)
            c = c >> jnp.uint64(32)
        c1 = c.astype(jnp.uint32)
        u = [None] * 8
        c = jnp.zeros((B,), dtype=jnp.uint64)
        for i in range(8):
            addc = jnp.uint64(0x3D1) if i == 0 else (jnp.uint64(1) if i == 1 else jnp.uint64(0))
            c = c + t[i].astype(jnp.uint64) + addc
            u[i] = (c & MASK32).astype(jnp.uint32)
            c = c >> jnp.uint64(32)
        sel = jnp.uint32(0) - (c1 | c.astype(jnp.uint32))
        r = [jnp.where(sel != 0, u[i], t[i]) for i in range(8)]
        return jnp.stack(r, axis=1)

    def fe_sub(a, b):
        B = a.shape[0]
        ac = [a[:, i].astype(jnp.int64) for i in range(8)]
        bc = [b[:, i].astype(jnp.int64) for i in range(8)]
        t = [None] * 8
        borrow = jnp.zeros((B,), dtype=jnp.int64)
        for i in range(8):
            d = ac[i] - bc[i] - borrow
            t[i] = (d & jnp.int64(0xFFFFFFFF)).astype(jnp.uint32)
            borrow = (d >> 63) & 1
        m = jnp.uint32(0) - borrow.astype(jnp.uint32)
        mi = m.astype(jnp.int64)
        tc = [t[i].astype(jnp.int64) for i in range(8)]
        r = [None] * 8
        bo = jnp.zeros((B,), dtype=jnp.int64)
        for i in range(8):
            k = (jnp.int64(0x3D1) & mi) if i == 0 else ((jnp.int64(1) & mi) if i == 1 else jnp.zeros((B,), dtype=jnp.int64))
            d = tc[i] - k - bo
            r[i] = (d & jnp.int64(0xFFFFFFFF)).astype(jnp.uint32)
            bo = (d >> 63) & 1
        return jnp.stack(r, axis=1)

    def _fe_reduce(t16):
        # t16 : liste de 16 tableaux (B,) uint32 -> renvoie (B,8) uint32, résultat mod p
        B = t16[0].shape[0]
        c = jnp.zeros((B,), dtype=jnp.uint64)
        s = [None] * 8
        for i in range(8):
            cur = t16[8 + i].astype(jnp.uint64) * jnp.uint64(977) + t16[i].astype(jnp.uint64) + c
            if i > 0:
                cur = cur + t16[7 + i].astype(jnp.uint64)
            s[i] = (cur & MASK32).astype(jnp.uint32)
            c = cur >> jnp.uint64(32)
        top = c + t16[15].astype(jnp.uint64)
        acc = s[0].astype(jnp.uint64) + top * jnp.uint64(977)
        s[0] = (acc & MASK32).astype(jnp.uint32); acc = acc >> jnp.uint64(32)
        acc = acc + s[1].astype(jnp.uint64) + top
        s[1] = (acc & MASK32).astype(jnp.uint32); acc = acc >> jnp.uint64(32)
        for i in range(2, 8):
            acc = acc + s[i].astype(jnp.uint64)
            s[i] = (acc & MASK32).astype(jnp.uint32); acc = acc >> jnp.uint64(32)
        m = jnp.uint32(0) - acc.astype(jnp.uint32)
        acc2 = s[0].astype(jnp.uint64) + (jnp.uint64(0x3D1) & m.astype(jnp.uint64))
        s[0] = (acc2 & MASK32).astype(jnp.uint32); acc2 = acc2 >> jnp.uint64(32)
        acc2 = acc2 + s[1].astype(jnp.uint64) + (jnp.uint64(1) & m.astype(jnp.uint64))
        s[1] = (acc2 & MASK32).astype(jnp.uint32); acc2 = acc2 >> jnp.uint64(32)
        for i in range(2, 8):
            acc2 = acc2 + s[i].astype(jnp.uint64)
            s[i] = (acc2 & MASK32).astype(jnp.uint32); acc2 = acc2 >> jnp.uint64(32)
        acc3 = jnp.zeros((B,), dtype=jnp.uint64)
        u = [None] * 8
        for i in range(8):
            addc = jnp.uint64(0x3D1) if i == 0 else (jnp.uint64(1) if i == 1 else jnp.uint64(0))
            acc3 = acc3 + s[i].astype(jnp.uint64) + addc
            u[i] = (acc3 & MASK32).astype(jnp.uint32)
            acc3 = acc3 >> jnp.uint64(32)
        sel = jnp.uint32(0) - acc3.astype(jnp.uint32)
        r = [jnp.where(sel != 0, u[i], s[i]) for i in range(8)]
        return jnp.stack(r, axis=1)

    def fe_mul(a, b):
        B = a.shape[0]
        ac = [a[:, i].astype(jnp.uint64) for i in range(8)]
        bc = [b[:, i].astype(jnp.uint64) for i in range(8)]
        t = [jnp.zeros((B,), dtype=jnp.uint32) for _ in range(16)]
        for i in range(8):
            carry = jnp.zeros((B,), dtype=jnp.uint64)
            for j in range(8):
                cur = ac[i] * bc[j] + t[i + j].astype(jnp.uint64) + carry
                t[i + j] = (cur & MASK32).astype(jnp.uint32)
                carry = cur >> jnp.uint64(32)
            t[i + 8] = carry.astype(jnp.uint32)
        return _fe_reduce(t)

    def fe_sqr(a):
        return fe_mul(a, a)  # pas de carré dédié (optimisation possible, cf. docta.cpp fe_sqr) : simplicité > vitesse ici

    def fe_sqr_n(a, n):
        t = a
        for _ in range(n):
            t = fe_sqr(t)
        return t

    def fe_inv(a):
        # a^(p-2), même chaîne d'additions que docta.cpp / puzzle.cu (255 carrés + 15 produits)
        x2 = fe_mul(fe_sqr(a), a)
        x3 = fe_mul(fe_sqr(x2), a)
        x6 = fe_mul(fe_sqr_n(x3, 3), x3)
        x9 = fe_mul(fe_sqr_n(x6, 3), x3)
        x11 = fe_mul(fe_sqr_n(x9, 2), x2)
        x22 = fe_mul(fe_sqr_n(x11, 11), x11)
        x44 = fe_mul(fe_sqr_n(x22, 22), x22)
        x88 = fe_mul(fe_sqr_n(x44, 44), x44)
        x176 = fe_mul(fe_sqr_n(x88, 88), x88)
        x220 = fe_mul(fe_sqr_n(x176, 44), x44)
        x223 = fe_mul(fe_sqr_n(x220, 3), x3)
        t = fe_mul(fe_sqr_n(x223, 23), x22)
        t = fe_mul(fe_sqr_n(t, 5), a)
        t = fe_mul(fe_sqr_n(t, 3), x2)
        t = fe_mul(fe_sqr_n(t, 2), a)
        return t

    def fe_is_zero(a):
        return jnp.all(a == 0, axis=1)


def fe_from_int(x, n=1):
    """Entier Python -> tableau numpy (n,8) uint32 (limbes 32 bits, little-endian). Host seulement."""
    import numpy as np
    v = [((x >> (32 * i)) & 0xFFFFFFFF) for i in range(8)]
    return np.tile(np.array(v, dtype=np.uint32), (n, 1))


def fe_to_int(limbs_row):
    x = 0
    for i in range(8):
        x |= int(limbs_row[i]) << (32 * i)
    return x


# ============================================================ SHA-256 + RIPEMD-160 vectorisés (JAX)
def _install_hash_ops():
    global sha256_block_from_pubkey, sha256_compress, ripemd160_from_sha256, hash160_vec
    U32 = jnp.uint32

    def rotr(x, n):
        return ((x >> jnp.uint32(n)) | (x << jnp.uint32(32 - n))) & jnp.uint32(0xFFFFFFFF)

    def rotl(x, n):
        return ((x << jnp.uint32(n)) | (x >> jnp.uint32(32 - n))) & jnp.uint32(0xFFFFFFFF)

    SHA_K = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    ]
    SHA_H0 = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]

    def sha256_block_from_pubkey(limbs, par):
        """limbs:(B,8) uint32 (X, limbe7=mot le plus significatif) ; par:(B,) uint32 (0/1).
        Construit les 16 mots du bloc unique (33 octets + bourrage), comme hash160_fast()."""
        B = limbs.shape[0]
        v = [limbs[:, i] for i in range(8)]
        prefix = (jnp.uint32(2) + par) & jnp.uint32(0xFFFFFFFF)
        w = [None] * 16
        w[0] = ((prefix << jnp.uint32(24)) | (v[7] >> jnp.uint32(8))) & jnp.uint32(0xFFFFFFFF)
        for i in range(1, 8):
            w[i] = ((v[8 - i] << jnp.uint32(24)) | (v[7 - i] >> jnp.uint32(8))) & jnp.uint32(0xFFFFFFFF)
        w[8] = ((v[0] << jnp.uint32(24)) | jnp.uint32(0x00800000)) & jnp.uint32(0xFFFFFFFF)
        for i in range(9, 15):
            w[i] = jnp.zeros((B,), dtype=U32)
        w[15] = jnp.full((B,), 264, dtype=U32)
        return w

    def sha256_compress(w):
        B = w[0].shape[0]
        W = list(w) + [None] * 48
        for i in range(16, 64):
            s0 = rotr(W[i - 15], 7) ^ rotr(W[i - 15], 18) ^ (W[i - 15] >> jnp.uint32(3))
            s1 = rotr(W[i - 2], 17) ^ rotr(W[i - 2], 19) ^ (W[i - 2] >> jnp.uint32(10))
            W[i] = (W[i - 16] + s0 + W[i - 7] + s1) & jnp.uint32(0xFFFFFFFF)
        a, b, c, d, e, f, g, h = [jnp.full((B,), x, dtype=U32) for x in SHA_H0]
        for i in range(64):
            S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
            ch = (e & f) ^ ((~e & jnp.uint32(0xFFFFFFFF)) & g)
            t1 = (h + S1 + ch + jnp.uint32(SHA_K[i]) + W[i]) & jnp.uint32(0xFFFFFFFF)
            S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            t2 = (S0 + maj) & jnp.uint32(0xFFFFFFFF)
            h, g, f, e, d, c, b, a = g, f, e, (d + t1) & jnp.uint32(0xFFFFFFFF), c, b, a, (t1 + t2) & jnp.uint32(0xFFFFFFFF)
        return [(x + y) & jnp.uint32(0xFFFFFFFF) for x, y in zip(SHA_H0, [a, b, c, d, e, f, g, h])]

    _R1, _R2, _S1, _S2, _K1, _K2 = H._R1, H._R2, H._S1, H._S2, H._K1, H._K2

    def _fv(j, x, y, z):
        M = jnp.uint32(0xFFFFFFFF)
        if j < 16: return x ^ y ^ z
        if j < 32: return (x & y) | ((~x & M) & z)
        if j < 48: return (x | ((~y) & M)) ^ z
        if j < 64: return (x & z) | (y & ((~z) & M))
        return x ^ (y | ((~z) & M))

    def ripemd160_from_sha256(words8):
        B = words8[0].shape[0]
        M = jnp.uint32(0xFFFFFFFF)

        def bswap32(x):
            return (((x & jnp.uint32(0x000000FF)) << jnp.uint32(24)) |
                    ((x & jnp.uint32(0x0000FF00)) << jnp.uint32(8)) |
                    ((x & jnp.uint32(0x00FF0000)) >> jnp.uint32(8)) |
                    ((x & jnp.uint32(0xFF000000)) >> jnp.uint32(24))) & M

        X = [bswap32(w) for w in words8]
        X.append(jnp.full((B,), 0x80, dtype=U32))
        for _ in range(5):
            X.append(jnp.zeros((B,), dtype=U32))
        X.append(jnp.full((B,), 256, dtype=U32))
        X.append(jnp.zeros((B,), dtype=U32))

        a = jnp.full((B,), 0x67452301, dtype=U32); b = jnp.full((B,), 0xEFCDAB89, dtype=U32)
        c = jnp.full((B,), 0x98BADCFE, dtype=U32); d = jnp.full((B,), 0x10325476, dtype=U32)
        e = jnp.full((B,), 0xC3D2E1F0, dtype=U32)
        a2, b2, c2, d2, e2 = a, b, c, d, e
        for j in range(80):
            t = (a + _fv(j, b, c, d) + X[_R1[j]] + jnp.uint32(_K1[j // 16])) & M
            t = rotl(t, _S1[j]); t = (t + e) & M
            a, e, d, c, b = e, d, rotl(c, 10), b, t
            t2 = (a2 + _fv(79 - j, b2, c2, d2) + X[_R2[j]] + jnp.uint32(_K2[j // 16])) & M
            t2 = rotl(t2, _S2[j]); t2 = (t2 + e2) & M
            a2, e2, d2, c2, b2 = e2, d2, rotl(c2, 10), b2, t2
        h0 = jnp.full((B,), 0x67452301, dtype=U32); h1 = jnp.full((B,), 0xEFCDAB89, dtype=U32)
        h2 = jnp.full((B,), 0x98BADCFE, dtype=U32); h3 = jnp.full((B,), 0x10325476, dtype=U32)
        h4 = jnp.full((B,), 0xC3D2E1F0, dtype=U32)
        n1 = (h2 + d + e2) & M
        n2 = (h3 + e + a2) & M
        n3 = (h4 + a + b2) & M
        n4 = (h0 + b + c2) & M
        n0 = (h1 + c + d2) & M
        return [n0, n1, n2, n3, n4]

    def hash160_vec(limbs, par):
        """limbs:(B,8) uint32 (X), par:(B,) uint32 -> (B,8) uint32 (hash160 dans les 5 premiers
        limbes, little-endian ; les 3 derniers sont nuls — format compatible avec fe_*)."""
        B = limbs.shape[0]
        w = sha256_block_from_pubkey(limbs, par)
        H8 = sha256_compress(w)
        r5 = ripemd160_from_sha256(H8)
        cols = list(r5) + [jnp.zeros((B,), dtype=U32)] * 3
        return jnp.stack(cols, axis=1)


# ============================================================ tables EC (host, bigint Python)
def build_tables(half):
    """gx[i],gy[i] = i*G (i=1..half) ; sst = (2*half)*G. Calculé une seule fois en Python/bigint
    (comme build_tables() dans docta.cpp), puis converti en tableaux de limbes pour JAX."""
    import numpy as np
    gx = np.zeros((half, 8), dtype=np.uint32)
    gy = np.zeros((half, 8), dtype=np.uint32)
    cur = H.G
    for i in range(1, half + 1):
        gx[i - 1] = fe_from_int(cur[0], 1)[0]
        gy[i - 1] = fe_from_int(cur[1], 1)[0]
        if i < half:
            cur = H.point_add(cur, H.G)
    sst = H.point_add(cur, cur)  # cur = half*G ici
    sstx = fe_from_int(sst[0], 1)[0]
    ssty = fe_from_int(sst[1], 1)[0]
    return gx, gy, sstx, ssty


# ============================================================ noyau de recherche (un "groupe" = 2*HALF clés)
class Engine:
    """Porte group_step/search_block de docta.cpp : vectorisé sur B lanes indépendantes.
    Chaque appel à .step() avance TOUTES les lanes d'un groupe (2*HALF clés chacune)."""

    def __init__(self, half, B, targets, addr_mode):
        self.half = half
        self.B = B
        self.addr_mode = addr_mode
        gx, gy, sstx, ssty = build_tables(half)
        self.gx = jnp.asarray(gx)
        self.gy = jnp.asarray(gy)
        self.sstx = jnp.asarray(jnp.tile(sstx[None, :], (B, 1)))
        self.ssty = jnp.asarray(jnp.tile(ssty[None, :], (B, 1)))
        self.one_b8 = jnp.asarray(fe_from_int(1, B))
        self.i_vals = jnp.arange(1, half + 1, dtype=jnp.int32)
        if addr_mode:
            tgt = jnp.asarray(_np_stack_targets_hash(targets))
        else:
            tgt = jnp.asarray(_np_stack_targets_x(targets))
        self.targets = tgt  # (T,8) uint32
        self._step = jax.jit(self._make_step())

    def _match(self, cand):
        # cand: (B,8) -> (B,) bool : vrai si cand figure dans self.targets (comparaison exacte des 8 limbes)
        eq = jnp.all(cand[:, None, :] == self.targets[None, :, :], axis=-1)  # (B,T)
        return jnp.any(eq, axis=-1)

    def _make_step(self):
        half, B, addr_mode, one_b8 = self.half, self.B, self.addr_mode, self.one_b8

        def group(Cx, Cy):
            # ---- passe avant : pre[i] = produit cumulé des (gx[i]-Cx), i=1..half (avec garde zéro)
            def fwd(acc, gx_row):
                gx_i = jnp.broadcast_to(gx_row[None, :], (B, 8))
                d = fe_sub(gx_i, Cx)
                zi = fe_is_zero(d)
                d_safe = jnp.where(zi[:, None], one_b8, d)
                acc2 = fe_mul(acc, d_safe)
                return acc2, (acc2, zi)

            acc_final, (pre_stack, zero_stack) = jax.lax.scan(fwd, one_b8, self.gx)
            dS = fe_sub(self.sstx, Cx)
            zs = fe_is_zero(dS)
            dS_safe = jnp.where(zs[:, None], one_b8, dS)
            acc_final = fe_mul(acc_final, dS_safe)
            inv0 = fe_inv(acc_final)

            invd_half = fe_mul(inv0, pre_stack[half - 1])
            inv0 = fe_mul(inv0, dS_safe)
            Cnx = Cny = None
            if True:
                t = fe_sub(self.ssty, Cy)
                l = fe_mul(t, invd_half)
                cnx = fe_sub(fe_sub(fe_mul(l, l), Cx), self.sstx)
                t2 = fe_sub(Cx, cnx)
                cny = fe_sub(fe_mul(l, t2), Cy)
                Cnx = jnp.where(zs[:, None], Cx, cnx)   # cas dégénéré (rarissime) : on recalculera le point entier au besoin
                Cny = jnp.where(zs[:, None], Cy, cny)

            pre_ext = jnp.concatenate([one_b8[None, :, :], pre_stack[:half - 1]], axis=0)  # pre_ext[k] = pre[k] pour k=i-1

            def bwd(carry, xs):
                inv, found_flag, found_off = carry
                gx_row, gy_row, zi, pre_im1, i_val = xs
                gx_i = jnp.broadcast_to(gx_row[None, :], (B, 8))
                gy_i = jnp.broadcast_to(gy_row[None, :], (B, 8))
                invd = fe_mul(inv, pre_im1)
                d = fe_sub(gx_i, Cx)
                d_safe = jnp.where(zi[:, None], one_b8, d)
                inv_next = fe_mul(inv, d_safe)

                t1 = fe_add(gy_i, Cy)
                l1 = fe_mul(t1, invd)
                x3m = fe_sub(fe_sub(fe_mul(l1, l1), Cx), gx_i)

                t2 = fe_sub(gy_i, Cy)
                l2 = fe_mul(t2, invd)
                x3p = fe_sub(fe_sub(fe_mul(l2, l2), Cx), gx_i)

                if addr_mode:
                    ym = fe_sub(fe_mul(l1, fe_sub(x3m, Cx)), Cy)
                    parm = ym[:, 0] & jnp.uint32(1)
                    cand_m = hash160_vec(x3m, parm)
                    yp = fe_sub(fe_mul(l2, fe_sub(Cx, x3p)), Cy)
                    parp = yp[:, 0] & jnp.uint32(1)
                    cand_p = hash160_vec(x3p, parp)
                else:
                    cand_m, cand_p = x3m, x3p

                hit_m = self._match(cand_m) & (~zi)
                hit_p = self._match(cand_p) & (~zi) & (i_val != half)

                step_hit = hit_m | hit_p
                offset_now = jnp.where(hit_m, -i_val, jnp.where(hit_p, i_val, found_off))
                newly = step_hit & (~found_flag)
                found_off2 = jnp.where(newly, offset_now, found_off)
                found_flag2 = found_flag | step_hit
                return (inv_next, found_flag2, found_off2), None

            xs = (self.gx, self.gy, zero_stack, pre_ext, self.i_vals)
            init = (inv0, jnp.zeros((B,), dtype=bool), jnp.zeros((B,), dtype=jnp.int32))
            (_, found_flag, found_off), _ = jax.lax.scan(bwd, init, xs, reverse=True)

            # clé centrale elle-même (offset 0) — vérifiée separement par comparaison directe
            if addr_mode:
                parc = Cy[:, 0] & jnp.uint32(1)
                cand_c = hash160_vec(Cx, parc)
            else:
                cand_c = Cx
            hit_c = self._match(cand_c)
            found_off = jnp.where(hit_c & (~found_flag), jnp.int32(0), found_off)
            found_flag = found_flag | hit_c

            return Cnx, Cny, found_flag, found_off

        def step(Cx, Cy):
            return group(Cx, Cy)

        return step

    def run_group(self, Cx, Cy):
        return self._step(Cx, Cy)


def _np_stack_targets_x(targets):
    import numpy as np
    rows = [fe_from_int(t.x, 1)[0] for t in targets]
    return np.stack(rows, axis=0)


def _np_stack_targets_hash(targets):
    import numpy as np
    rows = [fe_from_int(t.x, 1)[0] for t in targets]  # t.x porte déjà le hash160 comme entier 160 bits
    return np.stack(rows, axis=0)


# ============================================================ pilote (CLI, boucle principale)
def parse_range(bits_or_range):
    if ':' in bits_or_range or '-' in bits_or_range:
        sep = bits_or_range.find(':')
        if sep < 0:
            sep = bits_or_range.find('-')
        a, b = bits_or_range[:sep], bits_or_range[sep + 1:]
        clean = lambda h: h[2:] if h.lower().startswith('0x') else h
        start, end = int(clean(a), 16), int(clean(b), 16)
        if start < 1 or end < start or end >= H.N:
            raise ValueError("plage invalide")
        return start, end
    bits = int(bits_or_range)
    if not (1 <= bits <= 255):
        raise ValueError("nombre de bits invalide (1-255)")
    return (1 << (bits - 1)), (1 << bits) - 1


def main():
    ap = argparse.ArgumentParser(description="Recherche de clé privée secp256k1 (portage JAX/TPU de docta.cpp).")
    ap.add_argument('target', nargs='?', help="clé publique compressée, adresse, ou fichier de cibles (une par ligne)")
    ap.add_argument('range', nargs='?', help="nombre de bits, ou plage hex début:fin")
    ap.add_argument('address', nargs='?', choices=['address'], help="indique que 'target' est une adresse (ou un fichier d'adresses)")
    ap.add_argument('-p', '--puzzle', type=int, default=None, help="puzzle Bitcoin n° (1-80) : remplace target/range/address")
    ap.add_argument('--half', type=int, default=128, help="taille de fenêtre HALF (défaut 128 ; doit être modeste pour limiter le temps de compilation)")
    ap.add_argument('-t', '--lanes', type=int, default=1024, help="nombre de lanes (lots indépendants) exécutées en parallèle (défaut 1024)")
    ap.add_argument('--mode', choices=['seq', 'random'], default='random', help="ordre des blocs (défaut aléatoire)")
    ap.add_argument('--max-seconds', type=float, default=None, help="arrêt automatique après N secondes")
    ap.add_argument('--selftest', action='store_true', help="valide le moteur JAX contre l'arithmétique de référence, puis quitte")
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    if not HAVE_JAX:
        print(f"JAX n'est pas disponible dans cet environnement : {_JAX_IMPORT_ERROR}")
        print("Ce script est prévu pour un notebook Kaggle avec accélérateur TPU (JAX est préinstallé).")
        sys.exit(1)
    _install_fe_ops()
    _install_hash_ops()
    print(f"JAX {jax.__version__} — périphériques : {jax.devices()}")

    addr_mode = False
    target_arg = args.target
    range_arg = args.range

    if args.puzzle is not None:
        if not (1 <= args.puzzle <= H.PUZZLE_MAX):
            print(f"Puzzle inconnu : 1 à {H.PUZZLE_MAX}."); sys.exit(1)
        target_arg = H.PUZZLE_ADDR[args.puzzle]
        range_arg = str(args.puzzle)
        addr_mode = True
        print(f"Puzzle n°{args.puzzle} : {target_arg}  (clé de {args.puzzle} bits)")
    else:
        if args.address == 'address':
            addr_mode = True
        if target_arg is None or range_arg is None:
            ap.print_help(); sys.exit(1)

    start, end = parse_range(range_arg)

    targets = []
    if os.path.isfile(target_arg):
        targets, lines, valid, rejected = H.parse_targets_file(target_arg, addr_mode)
        print(f"Fichier lu : {lines} lignes, {valid} valides, {rejected} rejetées.")
        if not targets:
            print("Aucune cible valide."); sys.exit(1)
    else:
        t = H.parse_single_target(target_arg, addr_mode)
        if t is None:
            print("Cible invalide (clé publique compressée, adresse P2PKH/bech32, ou fichier attendu)."); sys.exit(1)
        targets = [t]
        print("Cible unique :", H.ent_text(t))

    half = args.half
    B = args.lanes
    chunk_keys = 2 * half  # clés couvertes par un lot (une "lane") en un appel
    total = end - start + 1
    print(f"Plage : {start:#x} .. {end:#x}  ({total} clés, HALF={half}, lanes={B})")
    print("Compilation JAX en cours (premier appel) — peut prendre un moment...")

    eng = Engine(half, B, targets, addr_mode)

    rng = random.Random(args.seed)
    n_super = (total + B * chunk_keys - 1) // (B * chunk_keys)
    if args.mode == 'random':
        order = list(range(n_super))
        rng.shuffle(order)
    else:
        order = range(n_super)

    t0 = time.time()
    keys_done = 0
    found = False
    for super_idx in order:
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            print("Temps maximal atteint, arrêt."); break
        super_start = start + super_idx * B * chunk_keys
        if super_start > end:
            continue

        import numpy as np
        Cx_np = np.zeros((B, 8), dtype=np.uint32)
        Cy_np = np.zeros((B, 8), dtype=np.uint32)
        lane_kc = []
        for b in range(B):
            kc = super_start + b * chunk_keys + half
            lane_kc.append(kc)
            if kc > H.N - 1:
                kc = 1  # lane hors plage : calcule quelque chose d'inoffensif, ses résultats seront ignorés
            pt = H.point_mul(kc, H.G)
            Cx_np[b] = fe_from_int(pt[0], 1)[0]
            Cy_np[b] = fe_from_int(pt[1], 1)[0]
        Cx = jnp.asarray(Cx_np)
        Cy = jnp.asarray(Cy_np)

        _, _, found_flag, found_off = eng.run_group(Cx, Cy)
        found_flag = jax.device_get(found_flag)
        found_off = jax.device_get(found_off)
        keys_done += B * chunk_keys

        hit_lanes = [b for b in range(B) if found_flag[b]]
        for b in hit_lanes:
            key = lane_kc[b] + int(found_off[b])
            if key < 1 or key > H.N - 1:
                continue
            pt = H.point_mul(key, H.G)
            ok = False
            match_target = None
            for t in targets:
                if t.is_addr:
                    if int.from_bytes(H.hash160(H.pub33(pt)), 'big') == t.x:
                        ok = True; match_target = t; break
                else:
                    if pt[0] == t.x:
                        ok = True; match_target = t; break
            if ok:
                where = H.save_result(key, match_target, {})
                print(f"\n*** CLÉ TROUVÉE : {key:#x}  (décimal {key}) — enregistrée : {where or 'ÉCHEC ÉCRITURE, NOTE-LA !'} ***\n")
                found = True
            else:
                print(f"(vérification exacte négative pour la lane {b}, faux positif ignoré — ne devrait pratiquement jamais arriver)")
        if found:
            break

        elapsed = time.time() - t0
        rate = keys_done / elapsed if elapsed > 0 else 0
        pct = 100.0 * min(keys_done, total) / total
        print(f"\r{keys_done:,} clés testées ({pct:5.2f} %) — {rate:,.0f} clés/s — {elapsed:6.1f} s".replace(',', ' '), end='', flush=True)

    print()
    if not found:
        print("Clé non trouvée dans la plage parcourue (ou arrêt anticipé).")


# ============================================================ auto-test
def run_selftest():
    print("=== auto-test (sans JAX, arithmétique de référence Python) ===")
    pt1 = H.point_mul(1, H.G)
    assert H.addr_legacy(H.hash160(H.pub33(pt1))) == H.PUZZLE_ADDR[1]
    pt3 = H.point_mul(7, H.G)
    assert H.addr_legacy(H.hash160(H.pub33(pt3))) == H.PUZZLE_ADDR[3]
    print("  OK : clés connues des puzzles #1 et #3 -> bonnes adresses.")

    if not HAVE_JAX:
        print(f"JAX indisponible ({_JAX_IMPORT_ERROR}) : auto-test limité à la partie host ci-dessus.")
        return

    _install_fe_ops()
    _install_hash_ops()
    import numpy as np
    rng = random.Random(123)

    # ---- arithmétique de corps
    B = 33
    xs = [rng.randrange(1, H.P) for _ in range(B)]
    ys = [rng.randrange(1, H.P) for _ in range(B)]
    A = jnp.asarray(np.stack([fe_from_int(x, 1)[0] for x in xs]))
    Bv = jnp.asarray(np.stack([fe_from_int(y, 1)[0] for y in ys]))
    R = jax.device_get(fe_add(A, Bv))
    for i in range(B):
        assert fe_to_int(R[i]) == (xs[i] + ys[i]) % H.P
    R = jax.device_get(fe_sub(A, Bv))
    for i in range(B):
        assert fe_to_int(R[i]) == (xs[i] - ys[i]) % H.P
    R = jax.device_get(fe_mul(A, Bv))
    for i in range(B):
        assert fe_to_int(R[i]) == (xs[i] * ys[i]) % H.P
    R = jax.device_get(fe_inv(A))
    for i in range(B):
        assert fe_to_int(R[i]) == pow(xs[i], H.P - 2, H.P)
    print(f"  OK : fe_add/fe_sub/fe_mul/fe_inv (JAX) == bigint Python sur {B} cas aléatoires.")

    # ---- hash160 vectorisé
    ks = [rng.randrange(1, H.N) for _ in range(B)]
    limbs = np.zeros((B, 8), dtype=np.uint32)
    par = np.zeros((B,), dtype=np.uint32)
    expected = []
    for i, k in enumerate(ks):
        pt = H.point_mul(k, H.G)
        limbs[i] = fe_from_int(pt[0], 1)[0]
        par[i] = pt[1] & 1
        expected.append(H.hash160(H.pub33(pt)))
    got = jax.device_get(hash160_vec(jnp.asarray(limbs), jnp.asarray(par)))
    for i in range(B):
        h20 = b''.join(int(got[i, j]).to_bytes(4, 'little') for j in range(5))
        assert h20 == expected[i]
    print(f"  OK : hash160_vec (JAX) == btc_host.hash160 sur {B} clés aléatoires.")

    # ---- moteur complet, bout en bout : on force le puzzle #1 (clé=1) dans une petite plage
    half = 8
    B2 = 16
    target = H.parse_single_target(H.PUZZLE_ADDR[1], True)
    eng = Engine(half, B2, [target], True)
    # lane 0 couvre exactement [1 .. 2*half] -> kc = half -> la clé 1 est à l'offset 1-half
    start = 1
    Cx_np = np.zeros((B2, 8), dtype=np.uint32)
    Cy_np = np.zeros((B2, 8), dtype=np.uint32)
    lane_kc = []
    for b in range(B2):
        kc = start + b * 2 * half + half
        lane_kc.append(kc)
        pt = H.point_mul(kc, H.G)
        Cx_np[b] = fe_from_int(pt[0], 1)[0]
        Cy_np[b] = fe_from_int(pt[1], 1)[0]
    _, _, found_flag, found_off = eng.run_group(jnp.asarray(Cx_np), jnp.asarray(Cy_np))
    found_flag = jax.device_get(found_flag); found_off = jax.device_get(found_off)
    hits = [(b, lane_kc[b] + int(found_off[b])) for b in range(B2) if found_flag[b]]
    assert hits == [(0, 1)], hits
    print("  OK : recherche bout en bout (JAX) retrouve la clé 1 du puzzle #1 dans une petite plage forcée.")
    print("=== auto-test terminé avec succès ===")


if __name__ == '__main__':
    main()
