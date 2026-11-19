# Portage JAX/TPU de docta.cpp / puzzle.cu

Deux fichiers :
- `btc_host.py` — partie "host" en Python pur (aucune dépendance JAX) : arithmétique
  secp256k1 en entiers arbitraires, SHA-256/RIPEMD-160, adresses base58/bech32,
  table des puzzles Bitcoin 1-80, parsing de cibles, écriture du résultat.
- `puzzle_tpu.py` — le moteur JAX/TPU proprement dit (arithmétique de corps et
  hachage vectorisés, algorithme de fenêtre par lot) + la ligne de commande.

Lis d'abord le long commentaire en tête de `puzzle_tpu.py` : il explique
honnêtement pourquoi un TPU est un **mauvais choix matériel** pour ce calcul
(c'est une unité de multiplication de matrices, pas un processeur bit-à-bit),
ce qui a dû être simplifié pour que XLA l'accepte, et ce qui a été volontairement
laissé de côté (modes Sobol/Monte Carlo/pi/e/ln, voisinage `-v`, filtre de Bloom,
cas dégénéré C = ±i·G). Ce n'est pas le même programme que `docta.cpp`/`puzzle.cu`,
c'est une **réécriture** qui en garde l'idée centrale (inversion de corps par lots)
en l'adaptant à ce qu'un accélérateur à formes statiques peut exécuter.

---

## 1. Préparer le notebook Kaggle

1. Crée un notebook Kaggle (**Code > New Notebook**).
2. **Settings (panneau de droite) > Accelerator > TPU VM v3-8** (ou la version TPU
   proposée au moment où tu lis ceci — l'interface Kaggle change de temps en temps).
3. Mets les deux fichiers dans `/kaggle/working/` : soit en les ajoutant comme
   **Notebook > Add input > Upload**, soit (plus simple) en collant leur contenu
   dans une cellule avec `%%writefile` :

   ```python
   %%writefile btc_host.py
   # (coller ici tout le contenu de btc_host.py)
   ```
   ```python
   %%writefile puzzle_tpu.py
   # (coller ici tout le contenu de puzzle_tpu.py)
   ```

4. Vérifie que JAX voit bien le TPU :

   ```python
   import jax
   print(jax.__version__, jax.devices())
   ```

   Tu dois voir quelque chose comme `[TpuDevice(id=0, ...), ...]` (8 cœurs pour un
   v3-8/v5e-8). Si tu vois `[CpuDevice(...)]`, l'accélérateur TPU n'est pas activé
   (retourne à l'étape 2) ou JAX ne le détecte pas (redémarre le notebook après
   avoir changé l'accélérateur).

   Si `import jax` échoue ou si `jax.devices()` ne montre pas de TPU avec le
   client PJRT attendu, il arrive que Kaggle ait besoin d'un paquet TPU explicite :

   ```python
   !pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
   ```

   (à exécuter **avant** le premier `import jax`, puis redémarrer le kernel si tu
   l'avais déjà importé).

## 2. Toujours commencer par l'auto-test

Avant toute vraie recherche, dans une cellule :

```python
!cd /kaggle/working && python3 puzzle_tpu.py --selftest
```

Ça vérifie, sur de petits cas, que l'arithmétique JAX (corps Fp, SHA-256+RIPEMD-160,
algorithme de fenêtre) donne **exactement** les mêmes résultats que l'arithmétique
de référence en entiers Python, et que le moteur complet retrouve bien une clé
connue (puzzle #1) de bout en bout. Je l'ai moi-même validé pièce par pièce en
Python/numpy avant de l'écrire en JAX (je n'ai pas d'accès à un TPU ni même à
JAX pour exécuter ce fichier dans l'environnement où je l'ai écrit), mais
`--selftest` est la vérification qui compte : **fais-la tourner avant de faire
confiance au moindre résultat.** Si elle échoue, ne lance pas de recherche —
dis-le-moi avec le message d'erreur.

## 3. Lancer une recherche

Toujours depuis `/kaggle/working` :

```bash
# Puzzle Bitcoin n°40 (adresse et plage de bits automatiques)
python3 puzzle_tpu.py -p 40

# Clé publique compressée, plage en bits
python3 puzzle_tpu.py 02f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036 10

# Adresse Bitcoin, plage hexadécimale explicite début:fin
python3 puzzle_tpu.py 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH 1:fffff address

# Fichier de cibles (une adresse ou une pubkey par ligne)
python3 puzzle_tpu.py mes_cibles.txt 40 address
```

Options utiles (voir aussi `python3 puzzle_tpu.py --help`) :

| Option | Rôle | Conseil |
|---|---|---|
| `--half N` | largeur de fenêtre (2·N clés par "groupe") | 64-256 ; plus grand = moins d'appels mais compilation plus longue |
| `-t`, `--lanes N` | nombre de lots indépendants calculés en parallèle | commence à 512-2048 ; monte tant que ça tient en mémoire TPU et que ça accélère |
| `--mode {seq,random}` | ordre de parcours des blocs | `random` par défaut (comme `-r` dans le programme original) |
| `--max-seconds N` | arrêt automatique | pratique pour un premier essai chronométré |
| `--seed N` | graine de l'ordre aléatoire des blocs | reproductibilité |

**Le premier appel compile** (traçage XLA puis compilation TPU) : attends-toi à
une pause de quelques secondes à quelques minutes selon `--half`/`--lanes` avant
que les premiers débits (clés/s) ne s'affichent. Les appels suivants réutilisent
le code compilé.

### Mode "pubkey" vs mode "adresse"

- Sans l'argument `address` : la cible est une clé publique compressée (`02`/`03`
  + 64 caractères hex). Le filtre rapide compare directement la coordonnée X —
  c'est le chemin le moins coûteux.
- Avec `address` : la cible est une adresse `1...` (P2PKH) ou `bc1q...` (bech32
  P2WPKH, taproot/P2WSH non supportés, comme dans le code d'origine). Chaque
  candidat doit alors être haché (SHA-256 + RIPEMD-160) **avant** comparaison —
  c'est plus coûteux par clé mais, ironiquement, ce hachage bit-à-bit est plus
  proche de ce qu'un TPU fait bien que l'arithmétique modulaire 256 bits.

### Résultat

En cas de succès, le script écrit (et relit pour vérifier) `cle_trouvee.txt`
et `cle_trouvee_copie.txt` dans le répertoire courant — pense à les télécharger
depuis l'onglet **Output** du notebook, ou à les copier vers `/kaggle/working`
si ce n'est pas déjà le cas, avant de fermer la session.

## 4. Si ça ne marche pas / si c'est trop lent

- **`OOM` / erreur mémoire TPU** : réduis `--lanes` et/ou `--half`.
- **Compilation très longue** : réduis `--half` (le corps de boucle est fixe,
  mais un `--half` énorme augmente la taille des tableaux `pre[]` stockés).
- **Débit décevant** : c'est attendu, voir le commentaire en tête du fichier —
  un GPU (voire un gros CPU multi-cœur) est probablement plus adapté à ce calcul
  que ce TPU. Ce script est une démonstration de portage fonctionnel, pas une
  garantie de vitesse.
- **`jax.devices()` ne montre pas de TPU** : vérifie l'accélérateur dans les
  paramètres du notebook, redémarre le kernel, ou réinstalle `jax[tpu]` (§1).
- **Tu veux les modes manquants (Sobol, `-v` voisinage, etc.)** : dis-le-moi,
  ce sont des ajouts localisés (le point d'extension côté hôte est la fonction
  qui construit `order`/les centres de lot dans `main()` ; côté noyau JAX rien
  ne change).

## 5. Pour aller plus loin

Idées d'optimisation si l'auto-test passe et que tu veux pousser la vitesse :
- fusionner plusieurs groupes par appel JIT avec `jax.lax.scan` (actuellement
  un appel Python = un groupe = 2·`half` clés par lane, pour rester simple et
  limiter les risques d'erreur que je ne peux pas déboguer moi-même sans TPU) ;
- un carré de corps dédié (`fe_sqr`) au lieu de `fe_mul(a,a)` ;
- répartir les lanes sur les 8 cœurs TPU explicitement avec `jax.pmap`/`shard_map`
  plutôt que de laisser XLA les traiter comme un simple axe vectoriel.
