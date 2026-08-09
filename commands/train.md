# SCARA Diffusion Policy

De code leest de HDF5-bestanden rechtstreeks uit `scara_act/datasets`; er staat
dus geen tweede lokale kopie in `scara_diffusion_policy`.

## Trainen

Vanaf `/home/tomoya/scara_ws/src`:

```bash
python -u -m scara_diffusion_policy.train
```

De defaults zijn bedoeld als eerste volledige RTX 3090-run:

- 320 x 240 input, batch size 16, 8 dataloader-workers en AMP;
- 50 epochs, AdamW, 500 warm-upstappen en cosine learning-rate;
- horizon 32, 2 observaties en 16 uitgevoerde acties;
- DDIM met 8 inference-stappen en EMA-checkpoints;
- de ResNet18 wordt meegetraind. Voeg alleen voor een snellere, lichtere run
  `--freeze-backbone` toe.

`horizon=30` met `n_action_steps=15` lijkt logisch, maar past niet door de
originele temporal U-Net: na tweemaal halveren en weer verdubbelen worden
sequentielengtes 30 → 15 → 8 → 16, waardoor de skip-connections niet meer
dezelfde lengte hebben. `32/16` is daarom de dichtstbijzijnde veilige
2:1-configuratie en wijzigt de upstream architectuur niet.

Als 16 niet in het GPU-geheugen past:

```bash
python -u -m scara_diffusion_policy.train --batch-size 8
```

Een korte technische test is geen echte training:

```bash
python -u -m scara_diffusion_policy.train \
  --ckpt-dir /tmp/scara_diffusion_smoke \
  --num-epochs 1 --batch-size 1 --num-workers 0 \
  --image-width 64 --image-height 48 --freeze-backbone \
  --max-train-batches 2 --max-val-batches 1
```

## Resume

`training_state_last.pt` bevat model, EMA, optimizer, LR-scheduler, AMP-scaler,
epoch en random states. De checkpointmap bepaalt automatisch welke run wordt
hervat; de oorspronkelijke modelinstellingen worden uit `train_config.json`
gebruikt.

```bash
python -u -m scara_diffusion_policy.train --resume
```

`--num-epochs` is het totale eind-epoch, niet het aantal extra epochs:

```bash
python -u -m scara_diffusion_policy.train --resume --num-epochs 75
```

## Evalueren

Zonder `--live` of `--execute` gebruikt eval alleen een opgenomen HDF5-frame.
Hij toont de zestien voorspelde absolute targets, de demonstratietargets en de MAE:

```bash
python -m scara_diffusion_policy.eval --episode 0 --frame 100
```

Camera en robotstate lezen, zonder motion commands:

```bash
python -m scara_diffusion_policy.eval --live --chunks 20
```

Alleen deze expliciete vorm stuurt `MoveJ` naar TCS:

```bash
python -m scara_diffusion_policy.eval --execute --chunks 20
```

Iedere target wordt eerst gecontroleerd op fysieke jointlimits en maximale
30-Hz-delta. Bij een fout, timing-overrun of `Ctrl+C` probeert eval altijd
`halt` te sturen. Met bijvoorbeeld `--start-action 2 --action-count 4` worden
alleen posities 2 t/m 5 uit iedere voorspelde chunk gebruikt.

Binnen een chunk worden de 16 targets op absolute 30-Hz-deadlines verstuurd
(ongeveer 0,53 seconde).
Camera-observaties en de volgende inference gebeuren bewust tussen chunks,
zodat een blokkerende cameraread de commandocadans niet kan verstoren. Er is
dus wel een korte replan-pauze tussen twee chunks.

De huidige opnames bevatten enkele waarden buiten de jointlimits uit
`constants.py`; de trainer waarschuwt daarvoor en eval weigert zulke targets.
Controleer daarom de echte TCS-limieten voordat je `--execute` gebruikt, in
plaats van de beveiliging alleen ruimer te zetten.

## Waarom acties intern tussen -1 en 1 liggen

Dit zijn niet de fysieke SCARA-jointwaarden. Net als de originele Diffusion
Policy worden de minimum- en maximumwaarde van iedere joint uit de
trainingsepisodes lineair naar `[-1, 1]` gezet. Na inference vertaalt de policy
de waarden terug naar meters/radialen voordat eval ze toont of naar TCS stuurt.
Daarmee is `clip_sample=True` consistent; oude z-score-checkpoints zijn daarom
niet compatibel met deze v2-port.

## Kaggle/Vast-bundle

```bash
cd /home/tomoya/scara_ws/src
python -m scara_diffusion_policy.transfer.make_tgz
```

Controleer eerst zonder een groot archief te schrijven:

```bash
python -m scara_diffusion_policy.transfer.make_tgz --dry-run
```

De uitvoer staat in:

```text
scara_diffusion_policy/transfer/output/scara_diffusion_h32_a16.tgz
```

De volledige Kaggle-upload, cloud-to-cloud-download, achtergrondtraining,
resume en checkpoint-export staan in
[`transfer/README.md`](../transfer/README.md).
