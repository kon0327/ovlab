# QuIC: princip, integrace do OpenVLA a plán ověření v OVLAB

**Verze:** 1.0, 2026-10-01  
**Účel:** pracovní technická specifikace pro disertaci a implementaci.  
**Status:** matematické jádro vychází z publikovaného QuIC; integrační kroky jsou návrh, nikoli zpráva o dokončené implementaci.  
**Související dokument:** [QuIC-WC.md](QuIC-WC.md).

## 1. Vymezení metody

QuIC (*Quantum-Inspired Compound Adapters*) je metoda parameter-efficient fine-tuning (PEFT). Zmrazí předtrénované váhy a učí kompaktní multiplikativní adaptér složený z ortogonálních compound matic. Vyšší compound řády jsou odvozeny ze stejné malé základní matice a nemají vlastní nezávislé parametry. [Q1, §3]

**Primární úspora se týká trénovatelných parametrů, optimalizačních stavů a uloženého adaptéru.** Původní váhy zůstávají potřebné pro inference. QuIC proto samo o sobě nezmenšuje hustý backbone checkpoint; případné sloučení adaptéru do vah zachová rozměry váhových matic. Násobné úspory adaptéru v článku nelze interpretovat jako stejné zmenšení celého OpenVLA. [Q1, §2–4]

Kvantová inspirace vychází z transformací zachovávajících Hammingovu váhu. Pro nízké compound řády lze metodu realizovat klasicky na CPU/GPU; OVLAB nepotřebuje kvantový počítač. Z publikovaných jazykových a vizuálních experimentů neplyne ověřená účinnost pro robotickou politiku. [Q1]

| Otázka | QuIC-PEFT | QuIC-WC |
|---|---|---|
| Co se učí nebo aproximuje? | Multiplikativní adaptér nad zmrazenou maticí | Kompaktní reprezentace samotné matice |
| Potřebuje nahrazovanou původní matici při inference? | Ano, případně její hustou sloučenou verzi | Ne; zůstávají faktory |
| Hlavní cíl | Levná adaptace | Menší samostatný model |
| Úspora inference VRAM | Není automatická; závisí i na backbone a kvantizaci | Cíl návrhu, který musí prokázat měření |
| Stav evidence | Publikovaná PEFT metoda | Navrhovaná výzkumná větev |

## 2. Matematický princip

### 2.1 Multiplikativní adaptace

Použijme konvenci sloupcových vektorů:

$$
W_0\in\mathbb R^{d_{out}\times d_{in}},\qquad y=W_0x+b_0.
$$

Výstupní QuIC adaptér má tvar

$$
W_\theta=R_\theta W_0,\qquad
R_\theta\in\mathbb R^{d_{out}\times d_{out}},\qquad
y=R_\theta(W_0x)+b_0.
$$

To rozšiřuje čtvercovou formulaci článku na běžné obdélníkové lineární vrstvy. Adaptér se konstruuje podle **výstupního rozměru** vrstvy. V tomto návrhu se bias nemění: otočení celého výsledku `Linear(x)` by současně otočilo i bias a implementovalo jinou operaci. U bezbiasových vrstev rozdíl odpadá.

V PyTorch se vstupy obvykle ukládají po řádcích. Ekvivalentní výpočet je `z = F.linear(x, W0, None)`, `z = z @ R.T`, následovaný přičtením biasu. Násobení zprava, $W_0R$, by bylo jinou, vstupní variantou; musí mít samostatné označení a testy.

### 2.2 Compound matice

Pro malou matici $A\in\mathbb R^{n\times n}$ a řád $k$ je

$$
C_k(A)\in\mathbb R^{\binom nk\times\binom nk},\qquad
[C_k(A)]_{I,J}=\det(A_{I,J}),
$$

kde $I,J$ jsou vzestupně uspořádané $k$-prvkové podmnožiny indexů. Je nutné fixovat jejich lexikografické pořadí, protože určuje mapování souřadnic. Platí $C_1(A)=A$.

Například prvek druhého řádu je

$$
[C_2(A)]_{(i,j),(p,q)}=A_{ip}A_{jq}-A_{iq}A_{jp}.
$$

Compound transformace tedy není libovolná nová velká matice: její prvky jsou svázané determinanty malé základní matice. Matematicky odpovídá působení $A$ na vnější mocnině prostoru. [Q1, §3.1]

### 2.3 Bloky a doplnění identitou

Nechť $d=d_{out}$, počet bloků je $N$ a velikost bloku $b=d/N$. Pro každý blok použijeme

$$
R_i=C_1(A_i)\oplus C_2(A_i)\oplus\cdots\oplus C_K(A_i)
\oplus I_{b-d_{comp}},
$$

$$
d_{comp}=\sum_{k=1}^{K}\binom nk\le b,\qquad
R=\bigoplus_{i=1}^{N}R_i.
$$

Stejná matice $A_i$ vytváří všechny řády uvnitř bloku. Volitelně lze sdílet $A_i$ také mezi bloky. Identitní doplnění je nezávislé na parametrech a dané souřadnice neadaptuje. Podíl takových souřadnic se musí reportovat; malé $n$ může znamenat především velkou neadaptovanou část.

Pro MVP požadujeme `d % N == 0`. Nerovnoměrné bloky jsou případné rozšíření, nikoli tiché zaokrouhlení rozměrů. Dokument používá jednoznačně $N$ pro počet bloků a $b$ pro velikost bloku; zdrojový článek místy používá označení `b` nejednotně. [Q1, rovnice 1–2 a obr. 3]

### 2.4 Ortogonalita a inicializace

Malou základní matici parametrizujeme Cayleyho transformací:

$$
S=-S^T,\qquad A=(I+S)(I-S)^{-1}.
$$

Ukládáme pouze nezávislé prvky nad diagonálou $S$. Při nulové inicializaci platí $A=I$, všechny compound matice jsou identity a $R=I$. Nově vložený adaptér tedy reprodukuje původní vrstvu. Inverzi implementujeme lineárním solve; konstrukci a determinanty nejprve počítáme ve FP32 a až výsledek převádíme do pracovního dtype. FP64 slouží pro numerické testy.

Z Cauchyho–Binetovy identity plyne

$$
C_k(A)^TC_k(A)=C_k(A^TA)=I.
$$

Ortogonalita se přenese na compound matice i na blokový adaptér. Pro výstupní variantu navíc

$$
(RW_0)^T(RW_0)=W_0^TW_0.
$$

Zachovávají se singulární hodnoty a Gramova matice sloupců. **Tato vlastnost nezaručuje zachování celé politiky:** změna souřadnic před nelinearitami, reziduálními vazbami a predikcí tokenů může změnit chování. Cayleyho parametrizace pokrývá část $SO(n)$ bez vlastní hodnoty $-1$; u identity je to přirozené lokální omezení. [Q1, příloha B]

### 2.5 Parametry a výpočetní náklady

Při minimální skew-symetrické parametrizaci je počet nezávislých parametrů

$$
P_{train}=N\frac{n(n-1)}2,
\qquad
P_{train,shared}=\frac{n(n-1)}2.
$$

Pokud implementace ukládá plnou pomocnou matici, počet skutečně alokovaných skalárů bude vyšší. Report musí odlišit nezávislé stupně volnosti, `Parameter.numel()` a bajty artefaktu.

Příklad pro **jednu** vrstvu s $d_{out}=4096$, $N=8$, $b=512$, bez sdílení:

| Compound sada | Největší přípustné $n$ | $d_{comp}$ na blok | Identitní doplnění na blok | Nezávislé parametry |
|---|---:|---:|---:|---:|
| $C_1$ | 512 | 512 | 0 | 1 046 528 |
| $C_1\oplus C_2$ | 31 | 496 | 16 | 3 720 |
| $C_1\oplus C_2\oplus C_3$ | 14 | 469 | 43 | 728 |

Jde o počet adaptérových parametrů, nikoli velikost celé vrstvy nebo modelu.

Blokové násobení má horní odhad $O(d^2/N)$ na vektor; hustý backbone výpočet zůstává. Přímá konstrukce compound matic stojí přibližně

$$
O\!\left(N\sum_{k=1}^{K}\binom nk^2 k^3\right)
$$

při běžném výpočtu determinantů. Ve frozen inference lze matice předpočítat. Během tréninku se po změně parametrů musí obnovit a zachovat gradientový graf; „jednorázová konstrukce“ proto neznamená jednou za celý trénink. Cache rozvinutých compound matic může být mnohem větší než uložené parametry adaptéru. [Q1, lemmata 2–3; praktická interpretace pro OVLAB]

## 3. Výhody a nevýhody

| Oblast | Výhoda | Omezení a co měřit |
|---|---|---|
| Adaptace | Velmi malý počet učitelných parametrů | Zmrazený backbone stále spotřebovává paměť a výpočet |
| Struktura | Přesně definované vazby mezi compound řády | Méně volnosti než nezávislá velká matice |
| Ortogonalita | Zachování norem a spektra samotné lineární mapy | Není důkaz action-safe chování |
| Inicializace | Identita umožňuje test ekvivalence bez adaptace | Mnoho identitního doplnění omezuje pokrytí |
| Checkpoint | Malý přenositelný adaptér | Samostatný adaptér není samostatná politika |
| Inference | Lze předpočítat adaptér nebo jej hustě sloučit | Rozvinutí přidává paměť; merge nekomprimuje backbone |
| Implementace | Nízké řády lze počítat klasicky | Determinanty, dtype, cache a autograd vyžadují ověření |
| Evidence | Výsledky na LLM a vizuálních úlohách | Výsledky na OpenVLA/LIBERO musí vzniknout experimentálně |

V hlavní větvi zachováme $C_1$. Původní ablace ukazuje výrazné zhoršení konfigurací bez prvního řádu; vyšší řády proto přidáváme k němu. Výměna determinantů za maximum nebo průměr není ekvivalentní QuIC konstrukce. [Q1, §5]

## 4. Integrace do OpenVLA

### 4.1 Identita hostitelského modelu

Základní OpenVLA propojuje DINOv2/SigLIP, multimodální projektor a jazykový backbone a vytváří akční tokeny. Aktuální disertační návrh lokalizuje action-proximal oblast do pozdních Llama vrstev a výstupní logitové cesty; deterministický detokenizér není trainable head. [Q2, §5.3]

Je nutné odlišovat:

- **OFT-orthogonal:** Orthogonal Fine-Tuning, matematický PEFT comparator QuIC; $C_1$ bez doplnění a se shodnými bloky jej reprodukuje.
- **OpenVLA-OFT:** Optimized Fine-Tuning, samostatná optimalizovaná VLA rodina. Ve zdokumentované OVLAB konfiguraci používá LoRA-adaptovaný backbone, kontinuální regresní head a akční chunky. Není reprodukcí ortogonálního OFT. [Q4, Q5]

Hlavní QuIC experiment doporučujeme provést na tokenovém OpenVLA. OpenVLA-OFT může být další hostitel, ale jeho vlastní head, vstupní modality a chunking se evidují samostatně.

### 4.2 Výběr lineárních vrstev

Vložení řídí explicitní seznam plných názvů modulů a jejich rozměrů získaný z `named_modules()`. Obvyklé názvy `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` jsou kandidáti k ověření, nikoli potvrzená mapa aktuálního lokálního checkpointu. V tomto workspace není zdrojový strom OVLAB ani načtený model.

| Region | První návrh zásahu | Hlavní riziko |
|---|---|---|
| Multimodální projektor | Jedna vybraná lineární vrstva, QP2; porovnat QP1 | Změna vizuálně-jazykového zarovnání |
| Střední Llama bloky | Nejprve `o_proj` nebo `down_proj`, poté další projekce | Zhoršení instrukčního a situačního zpracování |
| DINOv2/SigLIP | Až po fungující projekční/backbone větvi; každou větev evidovat | Ztráta vizuální informace |
| Pozdní Llama a logity | Ve výchozím běhu chráněné; QP1 jen jako řízená ablace | Přímá změna distribuce akčních tokenů |
| Detokenizér, normalizační statistiky | Beze změny | Jinak by se změnil samotný akční kontrakt |

Výstupní rotace nad `lm_head` působí v prostoru slovníku a může míchat akční i neakční tokenové souřadnice. Nelze ji automaticky nasadit jako běžný malý „7D action head“. Řádkově omezený zásah do akčních tokenů nebo vstupní rotace jsou samostatné rozšíření a musí být explicitně specifikovány.

### 4.3 Hostitel a férová reference

Podle současného PDF je praktickou referencí suite-matched LIBERO-adaptovaná politika; raw OpenVLA slouží jako architektonická či zero-shot reference. [Q2, §4.7.3]

Oddělíme dvě otázky:

1. **Srovnání metod adaptace:** QuIC, LoRA a OFT-orthogonal začínají ze stejného checkpointu a používají stejné demonstrace, preprocessing, tréninkový rozpočet a tokenovou architekturu.
2. **Přídavná adaptace hotové politiky:** QuIC se vloží nad zmrazený suite-specific checkpoint. Reference bez QuIC musí absolvovat srovnatelnou kontrolu dodatečného tréninku; jinak nelze připsat rozdíl pouze adaptéru.

Použití hotového LoRA checkpointu jako hostitele neprokazuje, že QuIC při původním tréninku LoRA nahradil. U checkpointu ověříme, zda jsou LoRA váhy sloučené, nebo aktivní za běhu, a zabráníme dvojímu merge.

### 4.4 Profily a umístění z aktuální disertace

Současný PDF návrh **nepoužívá QP0 jako QuIC profil**. Přebíráme QP1–QP4 a placement A1–A8; starší označení A0–A4 není zde výchozí. [Q2, tab. 4.7–4.8]

| Profil | Compound sada | Sdílení | Interpretace |
|---|---|---|---|
| QP1 | $C_1$ | Ne | Konzervativní profil ekvivalentní blokovému OFT při plném pokrytí |
| QP2 | $C_1\oplus C_2$ | Ne | Hlavní QuIC větev |
| QP3 | $C_1\oplus C_2\oplus C_3$ | Ne nebo ano | Silnější omezení parametrizace |
| QP4 | $C_1\oplus C_2\oplus C_3$ | Ano | Stress test s malým počtem parametrů; reportovat pokrytí |

Všechny hlavní profily vynucují ortogonalitu a determinantovou konstrukci. Konkrétní $N,n$ určuje rozměr vrstvy a musí být uloženy v manifestu. V původním značení znamená `gamma=0` zapnutou ortogonalitu a `beta=1` sdílení; v konfiguraci doporučujeme srozumitelné booleany.

**Profily nejsou automaticky monotónní škála komprese.** Počet parametrů závisí současně na $N,n,K$; vyšší $N$ ani vyšší řád nemají samy o sobě zaručený systémový efekt. Každou instanci rozhoduje skutečný počet parametrů, pokrytí a změřené náklady.

| Placement | Vision | Projektor | Language | Action-proximal |
|---|---|---|---|---|
| A1 | QP2–QP4 | Frozen | Frozen | Frozen |
| A2 | Frozen | QP1–QP3 | Frozen | Frozen |
| A3 | Frozen | Frozen | QP2–QP4 | Frozen |
| A4 | Frozen | Frozen | Frozen | QP1 |
| A5 | Frozen | QP2–QP3 | QP2–QP3 | Frozen |
| A6 | Frozen | QP2–QP3 | QP2–QP3 | QP1 |
| A7 | QP2–QP4 | QP2–QP3 | QP2–QP4 | QP1 |
| A8 | QP2–QP3 | Frozen | Frozen | QP1 |

Language a action-proximal masky musí být disjunktní. Nejprve realizujeme jednu A2 instanci, pak úzkou A3 instanci; úplný kartézský součin není nutný pro první důkaz proveditelnosti.

## 5. Návrh implementace v OVLAB

### 5.1 Odpovědnosti modulů

Navržené názvy jsou rozhraní k dopracování, nikoli existující API:

| Komponenta | Odpovědnost |
|---|---|
| `CompoundOperator` | Indexy minorů, diferenciabilní $C_k$, aplikace na aktivace |
| `OrthogonalBase` | Skew-symetrické parametry a Cayley solve |
| `QuICLinear` | Zmrazené váhy, blokové transformace, přesná práce s biasem |
| `QuICInjector` | Validace masky, náhrada vybraných vrstev, seznam trainable parametrů |
| `QuICArtifact` | Parametry adaptéru, konfigurace, reference a hash hostitele |
| `OpenVLAQuICPolicy` | Načtení přes existující modelový kontrakt a oficiální inference cestu |

Výchozí MVP je blockwise výpočet bez konstrukce plné $d\times d$ matice. Pro $C_2$ lze použít explicitní vzorec; pro $C_3$ malý referenční determinantový backend a následně vektorovanou implementaci. U identity mohou být mnohé minory singulární, proto je nutné otestovat gradienty determinantového backendu právě v inicializaci; alternativou jsou explicitní polynomiální vzorce.

Zmrazení backbone parametrů neznamená `no_grad()` nad všemi jeho operacemi: pro adaptér v dřívější vrstvě je potřebná diferenciace přes navazující zmrazené vrstvy. Při tréninku se používá akční tokenová loss se správným maskováním instrukce a vizuálního prefixu. [Q2, §5.3; návrh integrace]

### 5.2 Procesní hranice

QuIC policy service běží ve vlastním připnutém prostředí nebo kontejneru a využívá existující generic policy boundary. Dochovaná zpráva Gate E dokumentuje AF_UNIX RPC pro OpenVLA-OFT; jde o historický integrační důkaz, nikoli aktuální ověření QuIC. [Q5]

Přes rozhraní se přenášejí jen povolené obrazy, instrukce, případná deklarovaná propriocepce a identifikátory. Reward, success predicate ani skrytý stav simulátoru nejsou vstupem politiky. OVLAB zůstává vlastníkem rolloutu, akčního codec, trace a metrik. Nezavádíme nezdokumentované změny preprocessingu kvůli adaptéru.

### 5.3 Navržená konfigurace

Schéma je ilustrační a není připraveným příkazem současného CLI:

```yaml
method: quic_peft
host:
  family: openvla_token
  checkpoint_id: suite_matched_openvla_checkpoint
  revision: required_immutable_revision
  adaptation_state: merged_or_unmerged_must_be_verified
placement:
  id: A2
  target_manifest: resolved_module_manifest.json
  protect_action_proximal: true
adapter:
  profile: QP2
  orientation: output
  orders: [1, 2]
  block_count: 8
  base_size_rule: largest_feasible_per_target
  orthogonal: true
  share_across_blocks: false
  bias_policy: unchanged
  construction_dtype: float32
training:
  freeze_host: true
  objective: action_token_cross_entropy
  dataset_split_manifest: required_train_validation_split
export:
  mode: adapter_only
  require_host_hash: true
```

`block_count: 8` je příklad, který musí projít rozměrovou kontrolou všech vybraných vrstev. Neznámá revision nebo nepotvrzený merge stav jsou nevalidní konfigurace, nikoli hodnoty k automatickému odhadu.

### 5.4 Ukládání a reload

Adaptérový artefakt obsahuje parametry, compound konfiguraci, indexové pořadí, target manifest, dtype, verze kódu a jednoznačný odkaz na hostitele. Celkový deployable objem zahrnuje i hostitele, processor, tokenizer a normalizační statistiky.

Podporujeme dva oddělené režimy: `adapter_only` pro malý adaptér s hostitelem a `merged_dense` pro samostatný hustý checkpoint. Ekvivalenci ověříme na stejných vstupech včetně finálních akcí. Po `eval()` lze cache předpočítat; změna parametrů, dtype nebo device ji musí invalidovat. Kvantizaci řešíme až po nekvantizované větvi; hustý merge do kvantizovaných vah obvykle vyžaduje novou kvantizaci a mění numerické chyby.

## 6. Evaluace

Použijeme LIBERO-Spatial, Object, Goal a Long/10 se suite-specific referencemi a stejnými počátečními stavy. Původní LIBERO článek rozlišuje také LIBERO-100; LIBERO-10 nesmí být automaticky vydáváno za celou tuto sadu. [Q2, Q3]

| Vrstva evidence | Co reportovat |
|---|---|
| Adaptace | Trainable parametry, optimizer bytes, adapter bytes, peak training VRAM, doba a počet tréninkových kroků |
| Celá inference | Celkový checkpoint včetně závislostí, loaded a peak VRAM, host RAM, cache bytes |
| Runtime | Cold load, warm-up, model p50/p95 latency, policy service, RPC a uzavřená smyčka |
| Chování | SR po úlohách a suite, intervaly nejistoty, invalid actions, flicker, smoothness, opakované no-op |

Pro normalizované akční vektory lze použít diagnostiky

$$
Smooth1=\frac1{T-1}\sum_{t=2}^{T}\|a_t-a_{t-1}\|_2,
\qquad
Smooth2=\frac1{T-2}\sum_{t=3}^{T}\|a_t-2a_{t-1}+a_{t-2}\|_2.
$$

Oddělíme translaci, rotaci a gripper; jejich jednotky nelze bez deklarované normalizace smíchat. Tyto indexy nejsou fyzikální jerk; při různých časových rozestupech doplníme časově normalizované diagnostiky. Flicker je změna binárního stavu gripperu s uvedenou konvencí a počtem kroků. Porušení pracovního prostoru se odvozuje z definované robotické trajektorie/stavu, nikoli jen z rozsahu akčního vektoru.

Před výběrem kandidáta stanovíme $\delta_{SR}$, meze action-level metrik a systémový rozpočet. Například požadujeme $SR\ge SR_{ref}-\delta_{SR}$ a splnění latence/paměti; intervaly nejistoty a párování počátečních stavů musí být součástí interpretace. Checkpoint vybíráme na validačních datech a validačních rolloutech; finální evaluaci nepoužíváme k ladění. Pro OpenVLA-OFT evidujeme počet akcí v chunku i počet skutečně vykonaných akcí, aby se nesrovnávala amortizovaná cena s latencí nové informace.

## 7. Co je třeba udělat a rozhodnout

### P0: před prvním tréninkem

- [ ] Inventarizovat aktuální OVLAB checkout; ověřit policy, training a artifact rozhraní.
- [ ] Fixovat hostitele, checkpoint revision, suite a merge stav LoRA.
- [ ] Exportovat mapu vrstev, rozměrů a action-proximal ochrany; nezaměnit obě OFT metody.
- [ ] Implementovat FP32 referenci $C_1,C_2$, Cayley solve a wrapper s nezměněným biasem.
- [ ] Ověřit nulovou inicializaci, orientaci násobení, obdélníkové rozměry, ortogonalitu a gradienty u identity.
- [ ] Ověřit, že optimizer mění pouze deklarované parametry a gradient prochází frozen backbone.
- [ ] Připravit train/validation split a oficiální akční tokenovou loss.

### P1: první důkaz proveditelnosti

- [ ] Spustit jednu A2/QP2 variantu, reload a krátký diagnostický rollout.
- [ ] Prokázat rozdíl mezi adapter bytes, celým modelem a runtime cache.
- [ ] Přidat malou A3 variantu a shodný OFT-orthogonal/$C_1$ comparator.
- [ ] Provést měření tréninkové i inference paměti před rozšířením profilu.
- [ ] Nastavit suite-specific reference, počet training seeds a evaluací podle rozpočtu a nejistoty.

### P2: vědecká evaluace

- [ ] Dodat LoRA/QuIC/OFT-orthogonal srovnání při stejných datech a samostatně při shodném počtu parametrů.
- [ ] Rozšířit compound řády, sdílení a placement podle výsledků, nikoli pouze názvu profilu.
- [ ] Změřit chráněnou action-proximal větev; QP1 není předem dokázané bezpečné nastavení.
- [ ] Kvantizaci evidovat jako samostatný faktor a ověřit reload/equivalence.
- [ ] Aktualizovat kapitolu disertace: přesně oddělit adapter compression od weight compression.

Otevřená rozhodnutí: output versus input varianta; skutečné pokrytí identitním doplněním; sdílení parametrů mezi bloky; backend compound výpočtu; rozsah cílových vrstev; vhodný tréninkový rozpočet; limity přijatelné degradace. Připravený dokument tyto volby strukturuje, ale nemění jejich status na experimentálně ověřená rozhodnutí.

## 8. Kritérium dokončení

QuIC integrace je funkční, pokud identitní varianta reprodukuje hostitele, lze ji trénovat a znovu načíst a běží přes stejný OVLAB kontrakt. Výzkumný přínos vznikne až při doloženém trade-off mezi adaptačním rozpočtem a chováním politiky. Samotný malý počet parametrů neprokazuje úsporu celého deployovaného modelu ani action safety.

## 9. Zdroje a původ rozhodnutí

- **[Q1]** Raj, S.; Coyle, B. *QuIC: Quantum-Inspired Compound Adapters for Parameter Efficient Fine-Tuning.* Přiložený PDF, arXiv:2502.06916v2, 2025-10-05. Hlavně §3, §5 a příloha B. [Paper](https://arxiv.org/abs/2502.06916v2).
- **[Q2]** Konečný, J. *PhDDiS_Jiri_Konecny_2026 (3)(1).pdf*, přiložená současná verze. Hlavně §4.5–4.7, tab. 4.7–4.8 a §5.3. Autorita pro aktuální experimentální plán; není důkazem dokončených běhů.
- **[Q3]** Liu, B. et al. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning.* Přiložený PDF, arXiv:2306.03310v2, NeurIPS 2023. [Paper](https://arxiv.org/abs/2306.03310).
- **[Q4]** Oficiální repozitáře [OpenVLA](https://github.com/openvla/openvla) a [OpenVLA-OFT](https://github.com/moojink/openvla-oft), ověřené 2026-10-01. Použity k ověření identity a obecných rozdílů modelových rodin; přesná lokální mapa modulů musí být teprve exportována.
- **[Q5]** Projektová zpráva *Vložený markdown.md*, 2026-07-28, Gate E. Dokládá historickou AF_UNIX integraci nativního OpenVLA-OFT a význam Optimized Fine-Tuning; tříkrokový smoke rollout neprokazuje task success.

Uvedené komponenty, konfigurace, prioritizace a akceptační kroky jsou návrh tohoto dokumentu. V tomto zpracování nebyl proveden trénink ani benchmark a nebyl upraven repozitář OVLAB.
