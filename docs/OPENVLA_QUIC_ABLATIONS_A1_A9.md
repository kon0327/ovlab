# OpenVLA QuIC: konfigurace a předání ablací A1–A9

> Metodické poznámky k výzkumným konfiguracím. Historické soubory profilů byly
> odstraněny při oddělení pluginů. Vlastní studie se skládá z
> `plugin://openvla-quic/train/default.yaml`; pracovní recepty patří do ignorovaných
> uživatelských konfigurací. Aktuální integrační příklad je
> [plugin pipeline smoke](../examples/experiments/plugin-pipeline-smoke.yaml).

## Účel dokumentu

Tento dokument shrnuje implementované a návrhové konfigurace QuIC variant
OpenVLA v projektu OVLAB. Zaměřuje se na:

- pojmenované QuIC profily QP1–QP4,
- placementy A1–A9,
- povolené kombinace regionů a profilů,
- předkonfigurované smoke varianty,
- společné host/checkpoint, data a tréninkové nastavení,
- validační a spouštěcí pravidla,
- omezení a nejasnosti aktuálního stavu.

**Důležitá poznámka:** základní návrh disertačního PDF popisoval převážně
A1–A8. A9 je místní rozšíření zdrojového projektu a je implementované i ve
současném OVLAB kódu. K úplnému výzkumnému porovnání všech A1–A9 a všech
povolených QP profilů by byly potřeba další explicitní experimentální profily.

## Pojmy

### QuIC

QuIC je PEFT metoda založená na determinantalových compound operatorech.
Používá ortogonální konstrukci a nahrazuje nebo obaluje vybrané dense vrstvy
hostitelského modelu.

- `method: quic` označuje trénink QuIC adaptéru.
- `operation: determinant` je podporovaná operační forma.
- `orthogonal: true` zapíná ortogonální bázi.
- `orientation: output` určuje orientaci transformace.
- Vybrané vrstvy mají QuIC parametry; ostatní hostitelské parametry zůstávají
  zmrazené.

### Regiony modelu

| Region | Část modelu | Typický selektor |
|---|---|---|
| `vision` | Vision backbone a attention projekce | `vision_backbone.*.attn.proj` |
| `projector` | Projektor vision reprezentace | moduly začínající `projector.` |
| `language` | Jazykové transformer vrstvy mimo action-proximal část | `language_model.model.layers.*` podle masky |
| `action` | Poslední jazykové vrstvy nebo lm head | posledních `action_proximal_last_n` vrstev |

Rozhraní pro klasifikaci regionů je implementované v
[quic_experiments.py](../plugins/_shared/openvla-support/src/ovlab_openvla_training/quic_experiments.py).
`language_model.lm_head` je automaticky klasifikován jako `action`.

### QP profily

Profily se definuje jako pojmenované množiny compound řádů a block konfigurace.

| Profil | Compound orders | `block_count` | Sdílení bloků | `base_size` |
|---|---|---:|---|---:|
| QP1 | `[1]` | 8 | Ne | největší proveditelný rozměr |
| QP2 | `[1, 2]` | 8 | Ne | největší proveditelný rozměr |
| QP3 | `[1, 2, 3]` | 16 | Ne | největší proveditelný rozměr |
| QP4 | `[1, 2, 3]` | 32 | Ano | `3` |

Skutečné rozložení:

```python
PROFILES = {
    "QP1": {
        "orders": [1],
        "block_count": 8,
        "share_across_blocks": False,
        "base_size": None,
    },
    "QP2": {
        "orders": [1, 2],
        "block_count": 8,
        "share_across_blocks": False,
        "base_size": None,
    },
    "QP3": {
        "orders": [1, 2, 3],
        "block_count": 16,
        "share_across_blocks": False,
        "base_size": None,
    },
    "QP4": {
        "orders": [1, 2, 3],
        "block_count": 32,
        "share_across_blocks": True,
        "base_size": 3,
    },
}
```

Význam profilů:

- **QP1:** konzervativní profil; ekvivalentní blokovému OFT při vhodném
  pokrytí.
- **QP2:** hlavní větev experimentu s `C1` a `C2`.
- **QP3:** silnější parametrizace s `C1`, `C2` a `C3`.
- **QP4:** stress profil s větším počtem bloků a sdílenými bloky; počty
  parametrů, skutečné pokrytí a náklady je nutné měřit.

QP profily nejsou automaticky monotónní škálou komprese. Počet adaptérních
parametrů závisí zároveň na dimenzi vrstvy `N`, rozměru podprostorů `n` a
počtu bloků `K`.

## Matice placementů A1–A9

Zmrazené regiony nejsou cíle QuIC. Pokud placement region deklaruje, musí být
konfigurace vždy skutečně vybraná alespoň jedna aktivní vrstva z tohoto regionu.

| Placement | Vision | Projektor | Language | Action-proximal | Povolené profily |
|---|---|---|---|---|---|
| A1 | Aktivní | Frozen | Frozen | Frozen | Vision: QP2, QP3, QP4 |
| A2 | Frozen | Aktivní | Frozen | Frozen | Projektor: QP1, QP2, QP3 |
| A3 | Frozen | Frozen | Aktivní | Frozen | Language: QP2, QP3, QP4 |
| A4 | Frozen | Frozen | Frozen | Aktivní | Action: QP1 |
| A5 | Frozen | Aktivní | Aktivní | Frozen | Projektor: QP2/QP3; language: QP2/QP3 |
| A6 | Frozen | Aktivní | Aktivní | Aktivní | Projektor/language: QP2/QP3; action: QP1 |
| A7 | Aktivní | Aktivní | Aktivní | Aktivní | Vision/language: QP2/QP3/QP4; projektor: QP2/QP3; action: QP1 |
| A8 | Aktivní | Frozen | Frozen | Aktivní | Vision: QP2/QP3; action: QP1 |
| A9 | Aktivní | Frozen | Aktivní | Frozen | Vision/language: QP2/QP3/QP4 |

Přesná implementovaná mapa:

```python
PLACEMENTS = {
    "A1": {"vision": ["QP2", "QP3", "QP4"]},
    "A2": {"projector": ["QP1", "QP2", "QP3"]},
    "A3": {"language": ["QP2", "QP3", "QP4"]},
    "A4": {"action": ["QP1"]},
    "A5": {
        "projector": ["QP2", "QP3"],
        "language": ["QP2", "QP3"],
    },
    "A6": {
        "projector": ["QP2", "QP3"],
        "language": ["QP2", "QP3"],
        "action": ["QP1"],
    },
    "A7": {
        "vision": ["QP2", "QP3", "QP4"],
        "projector": ["QP2", "QP3"],
        "language": ["QP2", "QP3", "QP4"],
        "action": ["QP1"],
    },
    "A8": {
        "vision": ["QP2", "QP3"],
        "action": ["QP1"],
    },
    "A9": {
        "vision": ["QP2", "QP3", "QP4"],
        "language": ["QP2", "QP3", "QP4"],
    },
}
```

Zdroj: [quic_experiments.py](../plugins/_shared/openvla-support/src/ovlab_openvla_training/quic_experiments.py).

## Společná konfigurace hostitele a dat

Aktuální smoke konfigurace používají LIBERO-10 a jediný finetuned merged dense
host OpenVLA.

```yaml
trainer:
  id: openvla
  implementation: openvla-quic

model:
  base_checkpoint: openvla-7b-finetuned-libero-10
  revision: "80970322773f81baa2e22fe495d0487b93a05cfa"
  processor: inherited
  adaptation_state: merged

dataset:
  ref: libero/libero_10
  preparation: openvla-rlds
  split: train
  partition:
    seed: 42
    validation_fraction: 0.2
    max_episodes_per_group: 2
    frame_stride: 8
    max_frames_per_episode: 4
    action_normalization: host
```

Základní tréninkový rozpočet je:

```yaml
training:
  seed: 7
  max_steps: 3
  per_device_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 0.0001
  precision: bf16
  gradient_checkpointing: true
  quantization: none
  validation_batches: 4
  normalization:
    unnorm_key: libero_10

checkpointing:
  save_strategy: final
  save_optimizer_state: false
  output_kind: adapter

resources:
  accelerator: cuda
  gpu_count: 1
  max_vram_gib: 32
  network: disabled
```

Profil je `openvla-libero10-quic-smoke.yaml` (historický recept).

Toto je integrace/smoke rozpočet, nikoliv vědecky rozumný finální tréninkový
budget. Při porovnávání variant musejí zůstat stejné host identity, preprocessing,
dataset partition, seed, optimizer, počet kroků a hardwarový kontrakt.

## Společná PEFT konfigurace

```yaml
training:
  mode: peft
  peft:
    method: quic
    placement:
      id: A3
      action_proximal_last_n: 4
      allow_action_proximal: false
    adapter:
      profile: QP2
      operation: determinant
      orders: [1, 2]
      block_count: 8
      base_size: largest_feasible
      share_across_blocks: false
      orthogonal: true
      construction_dtype: float32
      orientation: output
    max_trainable_parameters: 1000000
```

Poznámky:

- `base_size: largest_feasible` je v plošině považován za největší proveditelný
  rozměr konkrétní cílové vrstvy.
- `block_size` lze zadat místo nebo vedle `block_count`. V tom případě musí
  rozdělení přesně dělit výstupní dimenzi.
- `allow_action_proximal: true` je nutné explicitně povolit jen u placementů,
  které skutečně cílí action region.
- Bez explicitního povolení jsou action-proximal vrstvy chráněny.
- Jazykové a action-proximal masky musí být disjunktní.

## Hotové smoke varianty

### A1-QP2

```yaml
kind: training
id: openvla-libero10-quic-a1-qp2
extends: plugin://openvla/training/quic/libero10-a1-qp2.yaml

training:
  peft:
    placement:
      id: A1
    targets:
      - names:
          - vision_backbone.featurizer.blocks.8.attn.proj
          - vision_backbone.featurizer.blocks.12.attn.proj
          - vision_backbone.fused_featurizer.blocks.8.attn.proj
          - vision_backbone.fused_featurizer.blocks.12.attn.proj
        region: vision
```

Konfigurace:
`libero10-a1-qp2.yaml` (historický recept)

A1 změňuje pouze vision backbone. Projektor, jazyk a action-proximal části jsou
zmrazené.

### A3-QP2

```yaml
kind: training
id: openvla-libero10-quic-a3-qp2
extends: plugin://openvla/training/quic/libero10-a3-qp2.yaml

training:
  peft:
    placement:
      id: A3
    targets:
      - names:
          - language_model.model.layers.4.self_attn.o_proj
          - language_model.model.layers.12.self_attn.o_proj
          - language_model.model.layers.20.self_attn.o_proj
          - language_model.model.layers.27.self_attn.o_proj
        region: language
```

Konfigurace:
`libero10-a3-qp2.yaml` (historický recept)

A3 používá vybrané jazykové self-attention projekce, nikoliv action-proximal
vrstvy.

### A9-QP2

```yaml
kind: training
id: openvla-libero10-quic-a9-qp2
extends: plugin://openvla/training/quic/libero10-a9-qp2.yaml

training:
  peft:
    placement:
      id: A9
    targets:
      - names:
          - vision_backbone.featurizer.blocks.8.attn.proj
          - vision_backbone.featurizer.blocks.12.attn.proj
          - vision_backbone.fused_featurizer.blocks.8.attn.proj
          - vision_backbone.fused_featurizer.blocks.12.attn.proj
        region: vision
      - names:
          - language_model.model.layers.4.self_attn.o_proj
          - language_model.model.layers.12.self_attn.o_proj
          - language_model.model.layers.20.self_attn.o_proj
          - language_model.model.layers.27.self_attn.o_proj
        region: language
```

Konfigurace:
`libero10-a9-qp2.yaml` (historický recept)

A9 kombinuje vision a language regiony. Projektor a action-proximal vrstvy jsou
zmrazené.

## Smoke ablační suite

Implementovaná suite sloučí pouze tři QP2 varianty:

```yaml
schema_version: ovlab.training-ablation/v1
id: openvla-quic-placement-smoke
base_profile: ../../experiments/training/openvla-libero10-quic-smoke.yaml
factors:
  - training.peft.placement
  - training.peft.targets
variants:
  A1-QP2: ../../experiments/training/quic/libero10-a1-qp2.yaml
  A3-QP2: ../../experiments/training/quic/libero10-a3-qp2.yaml
  A9-QP2: ../../experiments/training/quic/libero10-a9-qp2.yaml
```

Soubor:
`openvla-quic-placement-smoke.yaml` (historický recept)

Varianta je výběr placementu i explicitních target selektorů. Suite nesmí měnit
hostitelskou identitu, preprocessing, datovou partition, seed, optimizer nebo
tréninkový rozpočet.

## Validační pravidla

Před spuštěním se ověřuje:

1. placement identifikátor musí být `custom` nebo `A1`–`A9`;
2. pojmenované profily musí odpovídat povolenému regionu podle placementu;
3. každý target musí být existující module z inventáře hostitelského modelu;
4. jeden modul nesmí být zvolen dvěma překrývajícími se selektory;
5. target musí být podporovaná QuIC vrstva typu dense `Linear`/`Conv2d`;
6. target musí přispívat k policy outputům;
7. action-proximal targety jsou zakázané, pokud není nastaveno
   `allow_action_proximal: true`;
8. placement musí vybrat všechny deklarované regiony;
9. pojmenované profily nesmějí porušovat vlastní omezení compound setu,
   `block_count`, sharing a ortogonality;
10. `block_size` musí dělit výstupní dimenzi target modulu.

Náhodné nebo překrývající se selektory nejsou automaticky deduplikovány.
Překryv je chybou.

## Spuštění

Pouze čistá validace konfigurace:

```bash
./ovlab train ablate validate \
  --suite configs/ablations/openvla-quic-placement-smoke.yaml
```

Naplánování a ověření immutable host checkpointu a datasetu:

```bash
./ovlab train ablate plan \
  --suite configs/ablations/openvla-quic-placement-smoke.yaml
```

Materializace do nového výstupního adresáře:

```bash
./ovlab train ablate materialize \
  --suite configs/ablations/openvla-quic-placement-smoke.yaml \
  --output /tmp/quic-placement-plan
```

Sekvenční spuštění variant:

```bash
./ovlab train ablate run \
  --suite configs/ablations/openvla-quic-placement-smoke.yaml \
  --output /tmp/quic-placement-run
```

Další obecné instrukce jsou uvedené v
[ABLATIONS.md](ABLATIONS.md) a [QUIC_MIGRATION.md](QUIC_MIGRATION.md).

## Aktuální stav a omezení

### Co je implementované

- pojmenované QP1–QP4 profily,
- placementy A1–A9,
- resolver target modulů a regionů,
- validační kontroly profilů a míst,
- smoke konfigurace A1-QP2, A3-QP2 a A9-QP2,
- ABLATION suite pro tyto tři varianty,
- tréninkový OpenVLA QuIC runtime,
- immutable identity pro host, preprocessing, dataset partition a artefakty.

### Co zatím není implementované jako kompletní studie

- všechny povolené kombinace placement × QP profil,
- vědecké tréninkové rozpočty pro A1–A9,
- měření výsledné komprese a počtu parametrů,
- dokončené rollout porovnání,
- finální srovnání OpenVLA QuIC proti LoRA/OFT,
- plné cílené konfigurace pro A2, A4, A5, A6, A7 a A8.

### Co je nutné před finanční či vědeckou směrnou

1. Upevnit host checkpoint a jeho merged adaptation state.
2. Ověřit, že všechny varianty používají stejný seed, dataset partition,
   preprocessing a optimizer budget.
3. Pro každou variantu zapsat:
   - resolved target moduly,
   - profil,
   - skutečný počet trainable parametrů,
   - skutečný blokový rozklad,
   - VRAM a čas,
   - teacher-forced výsledky,
   - rollout výsledky.
4. Zkontrolovat, že language a action-proximal masky jsou disjunktní.
5. Nepoužít účinek obrázku nebo variance rolloutů jako důkaz trénovacího efektu
   bez společných tasků, stateů a seedů.

## Doporučený další postup

Pro první důkaz proveditelnosti je postupná sekvence vhodnější než úplný
kartézský součin:

1. Ověřit jednu A2/QP2 variantu na `projector` regionu.
2. Potvrdit úzkou A3 variantu na language regionu.
3. Použít A1/A3/A9 jako existující smoke třístupňovou integraci.
4. Následně rozšířit projektory a action-proximal cesty přes A5, A6, A7 a A8.
5. Až poté měřit QP3/QP4 a srovnávat skutečné parametry a pokrytí.

Tento postup snižuje počet kroků a umožňuje nejprve ověřit implementační cestu,
než se vytvoří úplná vědecká matrice.
