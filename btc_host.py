# btc_host.py — partie "host" en Python pur (aucune dépendance JAX).
#
# Rôle : tout ce qui n'a pas besoin d'accélération matérielle -> arithmétique
# EC de référence (via les entiers arbitraires de Python, pas de limbes),
# hachage, encodage/décodage d'adresses, parsing de cibles, écriture du
# résultat. Utilisé à la fois par puzzle_tpu.py (le moteur JAX/TPU) et par
# son --selftest.
#
# C'est la contrepartie directe des fonctions du même nom dans docta.cpp /
# puzzle.cu, mais réécrite avec les entiers natifs de Python (arbitraires),
# ce qui élimine tout le code U256 manuel du C++.

import hashlib
import os
import time

P = 2**256 - 2**32 - 977
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
G = (GX, GY)


# ---------------------------------------------------------------- secp256k1
def inv_mod(a, m):
    return pow(a, m - 2, m)


def point_add(P1, P2):
    if P1 is None:
        return P2
    if P2 is None:
        return P1
    x1, y1 = P1
    x2, y2 = P2
    if x1 == x2 and (y1 + y2) % P == 0:
        return None
    if P1 == P2:
        lam = (3 * x1 * x1) * inv_mod(2 * y1, P) % P
    else:
        lam = (y2 - y1) * inv_mod(x2 - x1, P) % P
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return (x3, y3)


def point_mul(k, Pt):
    R = None
    k = k % N
    while k:
        if k & 1:
            R = point_add(R, Pt)
        Pt = point_add(Pt, Pt)
        k >>= 1
    return R


def on_curve(pt):
    if pt is None:
        return False
    x, y = pt
    return (y * y - (x * x * x + 7)) % P == 0


def compress_hex(pt):
    x, y = pt
    return ('03' if y & 1 else '02') + format(x, '064x')


def x_from_compressed(s):
    """'02'/'03' + 64 hex -> (x, parity) si valide et sur la courbe, sinon None."""
    if len(s) != 66 or s[0] != '0' or s[1] not in '23':
        return None
    try:
        x = int(s[2:], 16)
    except ValueError:
        return None
    if x >= P:
        return None
    rhs = (x * x * x + 7) % P
    y = pow(rhs, (P + 1) // 4, P)
    if (y * y) % P != rhs:
        return None
    return x, (1 if s[1] == '3' else 0)


# ---------------------------------------------------------------- RIPEMD-160 (pur Python, 1 bloc)
_R1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
       3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
       4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13]
_R2 = [5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
       15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
       12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11]
_S1 = [11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
       11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
       9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6]
_S2 = [8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
       9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
       8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11]
_K1 = [0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E]
_K2 = [0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000]
_MASK = 0xFFFFFFFF


def _rotl(x, n):
    return ((x << n) | (x >> (32 - n))) & _MASK


def _f(j, x, y, z):
    if j < 16: return x ^ y ^ z
    if j < 32: return (x & y) | ((~x & _MASK) & z)
    if j < 48: return (x | ((~y) & _MASK)) ^ z
    if j < 64: return (x & z) | (y & ((~z) & _MASK))
    return x ^ (y | ((~z) & _MASK))


def ripemd160_block(X):
    a, b, c, d, e = 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
    a2, b2, c2, d2, e2 = a, b, c, d, e
    for j in range(80):
        t = (a + _f(j, b, c, d) + X[_R1[j]] + _K1[j // 16]) & _MASK
        t = _rotl(t, _S1[j])
        t = (t + e) & _MASK
        a, e, d, c, b = e, d, _rotl(c, 10), b, t

        t2 = (a2 + _f(79 - j, b2, c2, d2) + X[_R2[j]] + _K2[j // 16]) & _MASK
        t2 = _rotl(t2, _S2[j])
        t2 = (t2 + e2) & _MASK
        a2, e2, d2, c2, b2 = e2, d2, _rotl(c2, 10), b2, t2
    h = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0]
    t = (h[1] + c + d2) & _MASK
    h[1] = (h[2] + d + e2) & _MASK
    h[2] = (h[3] + e + a2) & _MASK
    h[3] = (h[4] + a + b2) & _MASK
    h[4] = (h[0] + b + c2) & _MASK
    h[0] = t
    return h


def ripemd160(msg: bytes) -> bytes:
    msg = bytearray(msg)
    ml = len(msg) * 8
    msg.append(0x80)
    while len(msg) % 64 != 56:
        msg.append(0)
    msg += ml.to_bytes(8, 'little')
    out = bytearray()
    for off in range(0, len(msg), 64):
        X = [int.from_bytes(msg[off + 4 * i:off + 4 * i + 4], 'little') for i in range(16)]
        H = ripemd160_block(X)
        for v in H:
            out += v.to_bytes(4, 'little')
    # NB: pour nos usages (33 ou 32 octets avant bourrage) un seul bloc suffit toujours.
    return bytes(out[-20:])


def hash160(data: bytes) -> bytes:
    return ripemd160(hashlib.sha256(data).digest())


def pub33(pt) -> bytes:
    x, y = pt
    return bytes([2 + (y & 1)]) + x.to_bytes(32, 'big')


# ---------------------------------------------------------------- base58check
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58check_encode(version: int, h20: bytes) -> str:
    payload = bytes([version]) + h20
    chk = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    data = payload + chk
    n = int.from_bytes(data, 'big')
    s = ''
    while n > 0:
        n, r = divmod(n, 58)
        s = B58[r] + s
    n_zeros = 0
    for byte in data:
        if byte == 0:
            n_zeros += 1
        else:
            break
    return '1' * n_zeros + s


def b58check_decode(s: str):
    """Retourne (code, h20) : code 0=invalide, 1=P2PKH, 2=valide mais non supporté."""
    if not s or len(s) > 40:
        return 0, None
    n_zeros = 0
    for ch in s:
        if ch == '1':
            n_zeros += 1
        else:
            break
    n = 0
    for ch in s:
        idx = B58.find(ch)
        if idx < 0:
            return 0, None
        n = n * 58 + idx
    nbytes = (n.bit_length() + 7) // 8
    data = b'\x00' * n_zeros + n.to_bytes(nbytes, 'big')
    if len(data) != 25:
        return 0, None
    payload, chk = data[:21], data[21:]
    if hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4] != chk:
        return 0, None
    if payload[0] != 0x00:
        return 2, None
    return 1, payload[1:]


# ---------------------------------------------------------------- bech32 (P2WPKH v0 uniquement)
B32 = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def _bech32_polymod(values):
    GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = ((chk & 0x1ffffff) << 5) ^ v
        for i in range(5):
            if (b >> i) & 1:
                chk ^= GEN[i]
    return chk


def _bech32_hrp_expand(hrp):
    return [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]


def bech32_encode_p2wpkh(h20: bytes) -> str:
    acc, bits = 0, 0
    data = [0]
    for byte in h20:
        acc = (acc << 8) | byte
        bits += 8
        while bits >= 5:
            bits -= 5
            data.append((acc >> bits) & 31)
    if bits:
        data.append((acc << (5 - bits)) & 31)
    values = _bech32_hrp_expand('bc') + data + [0, 0, 0, 0, 0, 0]
    pm = _bech32_polymod(values) ^ 1
    chk = [(pm >> (5 * (5 - i))) & 31 for i in range(6)]
    return 'bc1' + ''.join(B32[d] for d in data + chk)


def bech32_decode_addr(s0: str):
    """Retourne (code, h20) : 0=invalide, 1=P2WPKH v0, 2=valide mais non supporté (v1+/P2WSH)."""
    if len(s0) < 14 or len(s0) > 90:
        return 0, None
    has_lo = any(c.islower() for c in s0)
    has_up = any(c.isupper() for c in s0)
    if has_lo and has_up:
        return 0, None
    s = s0.lower()
    pos = s.rfind('1')
    if pos < 1 or len(s) - pos < 7:
        return 0, None
    hrp, rest = s[:pos], s[pos + 1:]
    data = []
    for c in rest:
        idx = B32.find(c)
        if idx < 0:
            return 0, None
        data.append(idx)
    pm = _bech32_polymod(_bech32_hrp_expand(hrp) + data)
    if pm not in (1, 0x2bc830a3) or hrp != 'bc':
        return 0, None
    data = data[:-6]
    if not data:
        return 0, None
    if data[0] != 0 or pm != 1:
        return 2, None  # taproot / segwit v1+ : non supporté ici
    acc, bits = 0, 0
    prog = bytearray()
    for v in data[1:]:
        acc = (acc << 5) | v
        bits += 5
        if bits >= 8:
            bits -= 8
            prog.append((acc >> bits) & 255)
    if bits >= 5 or ((acc << (8 - bits)) & 255):
        return 0, None
    if len(prog) != 20:
        return 2, None  # P2WSH : non supporté
    return 1, bytes(prog)


def addr_legacy(h20: bytes) -> str:
    return b58check_encode(0x00, h20)


def addr_bech32(h20: bytes) -> str:
    return bech32_encode_p2wpkh(h20)


def parse_address(s: str):
    """Retourne (code, h20): 0 invalide, 1 ok, 2 valide mais non supporté (P2SH/P2WSH/Taproot)."""
    code, h20 = b58check_decode(s)
    if code != 0:
        return code, h20
    return bech32_decode_addr(s)


# ---------------------------------------------------------------- cibles (pubkey ou adresse)
class Target:
    __slots__ = ('x', 'par', 'is_addr')

    def __init__(self, x, par, is_addr):
        self.x = x          # coord. X (mode pubkey) OU hash160 empaqueté en int 160 bits (mode adresse)
        self.par = par      # parité de Y (mode pubkey) ; 0 en mode adresse
        self.is_addr = is_addr


def parse_single_target(tok: str, addr_mode: bool):
    """Essaie de parser tok comme cible unique. Retourne un Target ou None."""
    tok = tok.strip()
    if not tok:
        return None
    if addr_mode:
        code, h20 = parse_address(tok)
        if code == 1:
            return Target(int.from_bytes(h20, 'big'), 0, True)
        return None
    r = x_from_compressed(tok)
    if r is None:
        return None
    x, par = r
    return Target(x, par, False)


def parse_targets_file(path: str, addr_mode: bool):
    """Une cible par ligne. Retourne (liste_de_Target, n_lignes, n_valides, n_rejetees)."""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    if text.startswith('\ufeff'):
        text = text[1:]
    targets, lines, rejected = [], 0, 0
    for line in text.splitlines():
        tok = line.strip().split()
        if not tok:
            continue
        lines += 1
        t = parse_single_target(tok[0], addr_mode)
        if t is not None:
            targets.append(t)
        else:
            rejected += 1
    # dédoublonnage
    seen = set()
    uniq = []
    for t in targets:
        key = (t.x, t.par)
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq, lines, len(uniq), rejected


def ent_text(t: Target) -> str:
    if t.is_addr:
        h20 = t.x.to_bytes(20, 'big')
        return f"{addr_legacy(h20)}  (bech32 {addr_bech32(h20)})"
    return ('03' if t.par else '02') + format(t.x, '064x')


# ---------------------------------------------------------------- puzzles Bitcoin connus (1..80)
# Challenge public et documenté depuis 2015 ("Bitcoin puzzle transaction") : des adresses
# financées dont la clé privée est garantie dans une plage de bits connue, à des fins de
# démonstration/compétition cryptographique. Rien à voir avec une cible privée arbitraire.
PUZZLE_MAX = 80
PUZZLE_ADDR = [None,
    "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH","1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb",
    "19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA","1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e",
    "1E6NuFjCi27W5zoXg8TRdcSRq84zJeBW3k","1PitScNLyp2HCygzadCh7FveTnfmpPbfp8",
    "1McVt1vMtCC7yn5b9wgX1833yCcLXzueeC","1M92tSqNmQLYw33fuBvjmeadirh1ysMBxK",
    "1CQFwcjw1dwhtkVWBttNLDtqL7ivBonGPV","1LeBZP5QCwwgXRtmVUvTVrraqPUokyLHqe",
    "1PgQVLmst3Z314JrQn5TNiys8Hc38TcXJu","1DBaumZxUkM4qMQRt2LVWyFJq5kDtSZQot",
    "1Pie8JkxBT6MGPz9Nvi3fsPkr2D8q3GBc1","1ErZWg5cFCe4Vw5BzgfzB74VNLaXEiEkhk",
    "1QCbW9HWnwQWiQqVo5exhAnmfqKRrCRsvW","1BDyrQ6WoF8VN3g9SAS1iKZcPzFfnDVieY",
    "1HduPEXZRdG26SUT5Yk83mLkPyjnZuJ7Bm","1GnNTmTVLZiqQfLbAdp9DVdicEnB5GoERE",
    "1NWmZRpHH4XSPwsW6dsS3nrNWfL1yrJj4w","1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum",
    "14oFNXucftsHiUMY8uctg6N487riuyXs4h","1CfZWK1QTQE3eS9qn61dQjV89KDjZzfNcv",
    "1L2GM8eE7mJWLdo3HZS6su1832NX2txaac","1rSnXMr63jdCuegJFuidJqWxUPV7AtUf7",
    "15JhYXn6Mx3oF4Y7PcTAv2wVVAuCFFQNiP","1JVnST957hGztonaWK6FougdtjxzHzRMMg",
    "128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k","12jbtzBb54r97TCwW3G1gCFoumpckRAPdY",
    "19EEC52krRUK1RkUAEZmQdjTyHT7Gp1TYT","1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps",
    "1LhE6sCTuGae42Axu1L1ZB7L96yi9irEBE","1FRoHA9xewq7DjrZ1psWJVeTer8gHRqEvR",
    "187swFMjz1G54ycVU56B7jZFHFTNVQFDiu","1PWABE7oUahG2AFFQhhvViQovnCr4rEv7Q",
    "1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb","1Be2UF9NLfyLFbtm3TCbmuocc9N1Kduci1",
    "14iXhn8bGajVWegZHJ18vJLHhntcpL4dex","1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2",
    "122AJhKLEfkFBaGAd84pLp1kfE7xK3GdT8","1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv",
    "1L5sU9qvJeuwQUdt4y1eiLmquFxKjtHr3E","1E32GPWgDyeyQac4aJxm9HVoLrrEYPnM4N",
    "1PiFuqGpG8yGM5v6rNHWS3TjsG6awgEGA1","1CkR2uS7LmFwc3T2jV8C1BhWb5mQaoxedF",
    "1NtiLNGegHWE3Mp9g2JPkgx6wUg4TW7bbk","1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP",
    "1Pd8VvT49sHKsmqrQiP61RsVwmXCZ6ay7Z","1DFYhaB2J9q1LLZJWKTnscPWos9VBqDHzv",
    "12CiUhYVTTH33w3SPUBqcpMoqnApAV4WCF","1MEzite4ReNuWaL5Ds17ePKt2dCxWEofwk",
    "1NpnQyZ7x24ud82b7WiRNvPm6N8bqGQnaS","15z9c9sVpu6fwNiK7dMAFgMYSK4GqsGZim",
    "15K1YKJMiJ4fpesTVUcByoz334rHmknxmT","1KYUv7nSvXx4642TKeuC2SNdTk326uUpFy",
    "1LzhS3k3e9Ub8i2W1V8xQFdB8n2MYCHPCa","17aPYR1m6pVAacXg1PTDDU7XafvK1dxvhi",
    "15c9mPGLku1HuW9LRtBf4jcHVpBUt8txKz","1Dn8NF8qDyyfHMktmuoQLGyjWmZXgvosXf",
    "1HAX2n9Uruu9YDt4cqRgYcvtGvZj1rbUyt","1Kn5h2qpgw9mWE5jKpk8PP4qvvJ1QVy8su",
    "1AVJKwzs9AskraJLGHAZPiaZcrpDr1U6AB","1Me6EfpwZK5kQziBwBfvLiHjaPGxCKLoJi",
    "1NpYjtLira16LfGbGwZJ5JbDPh3ai9bjf4","16jY7qLJnxb7CHZyqBP8qca9d51gAjyXQN",
    "18ZMbwUFLMHoZBbfpCjUJQTCMCbktshgpe","13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so",
    "1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9","1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ",
    "19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG","19YZECXj3SxEZMoUeJ1yiPsw8xANe7M7QR",
    "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU","1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR",
    "12VVRNPi4SJqUTsp6FmqDqY5sGosDtysn4","1FWGcVDK3JGzCC3WtkYetULPszMaK2Jksv",
    "1J36UjUByGroXcCvmj13U6uwaVv9caEeAt","1DJh2eHFYQfACPmrvpyWc8MSTYKh7w9eRF",
    "1Bxk4CQdqL9p22JEtDfdXMsng1XacifUtE","15qF6X51huDjqTmF9BJgxXdt1xcj46Jmhb",
    "1ARk8HWJMn8js8tQmGUJeQHjSE7KRkn2t8","1BCf6rHUW6m3iH2ptsvnjgLruAiPQQepLe",
]
assert len(PUZZLE_ADDR) == PUZZLE_MAX + 1


# ---------------------------------------------------------------- enregistrement du résultat
def append_checked(path: str, text: str) -> bool:
    try:
        with open(path, 'ab') as f:
            f.write(text.encode('utf-8'))
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        return False
    try:
        with open(path, 'rb') as f:
            data = f.read()
        return data.endswith(text.encode('utf-8'))
    except OSError:
        return False


def save_result(key_int: int, target: Target, extra: dict, out_main='cle_trouvee.txt',
                 out_copy='cle_trouvee_copie.txt') -> str:
    lines = []
    if target.is_addr:
        h20 = target.x.to_bytes(20, 'big')
        lines.append(f"Adresse (P2PKH)     : {addr_legacy(h20)}")
        lines.append(f"Adresse (bech32)    : {addr_bech32(h20)}")
        lines.append(f"Hash160             : {h20.hex()}")
    pt = point_mul(key_int, G)
    lines.append(f"Clé publique        : {compress_hex(pt)}")
    for k, v in extra.items():
        lines.append(f"{k:<20}: {v}")
    lines.append(f"Clé privée (hex)    : {key_int:064x}")
    lines.append(f"Clé privée (décimal): {key_int}")
    lines.append(f"Trouvée le          : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("-" * 40)
    text = "\n".join(lines) + "\n"
    used = []
    for p in (out_main, out_copy):
        if p and append_checked(p, text):
            used.append(p)
    if not used:
        alt = f"cle_trouvee_urgence_{int(time.time())}.txt"
        if append_checked(alt, text):
            used.append(alt)
    if not used:
        print("\n*** ATTENTION : clé trouvée mais IMPOSSIBLE À ENREGISTRER ! Note-la tout de suite : ***")
        print(text)
        return ""
    return " + ".join(used)


if __name__ == '__main__':
    # auto-test minimal : clés connues des puzzles Bitcoin publics #1 et #3 (solutions publiées)
    pt1 = point_mul(1, G)
    assert addr_legacy(hash160(pub33(pt1))) == PUZZLE_ADDR[1]
    pt3 = point_mul(7, G)
    assert addr_legacy(hash160(pub33(pt3))) == PUZZLE_ADDR[3]
    assert bech32_decode_addr(bech32_encode_p2wpkh(hash160(pub33(pt1))))[1] == hash160(pub33(pt1))
    print("btc_host.py : auto-test OK")
