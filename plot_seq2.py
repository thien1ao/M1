"""Erzeugt Images/seq2_cwagen_stromregelung.png für den Bericht (Sequenz 2)."""
import glob

import matplotlib.pyplot as plt
import polars as pl

SEQ = 1  # 0-basiert -> Sequenz 2
PAD = 3  # Sekunden vor/nach "Schweißen EIN"

EIN = '[72_1]S7-Schweißmaschine - Schweißen EIN (M215_0)'
FREIGABE = '[72_65]Schweißstromregelung: Reglerfeigabe (M2220_0)'
POS = '[72:46]S7-Schweißmaschine - Antrieb C-Wagen Istposition in mm (DB40_DBD32)'
VEL = '[72:47]S7-Schweißmaschine - Antrieb C-Wagen V-Ist in mm/s (DB40_DBD36)'
I_IST = '[72:70]Schweißstrom_HKS Zuführung (DB550_DBD0)'
I_SOLL = '[72:61]Schweißstromregelung: Schweißstrom Sollwert skaliert (DB510_DBD8)'
I_OUT = '[72:67]Schweißstromregelung: Reglerausgang (DB510_DBD28)'

BLUE, ORANGE, INK, MUTED = '#2a78d6', '#e8643c', '#333333', '#666666'

lf = pl.scan_parquet(sorted(glob.glob('./data/EV__*.parquet')))

seqs = (
    lf.select('Time', EIN).drop_nulls()
    .with_columns(seq_id=(pl.col(EIN) & ~pl.col(EIN).shift(fill_value=False)).cum_sum() - 1)
    .filter(pl.col(EIN))
    .group_by('seq_id').agg(start=pl.col('Time').min(), end=pl.col('Time').max())
    .filter(pl.col('seq_id') == SEQ)
    .collect(engine='streaming')
)
start, end = seqs.row(0)[1:]

df = (
    lf.filter(pl.col('Time').is_between(start - pl.duration(seconds=PAD), end + pl.duration(seconds=PAD)))
    .select('Time', FREIGABE, POS, VEL, I_IST, I_SOLL, I_OUT)
    .with_columns(t=(pl.col('Time') - start).dt.total_milliseconds() / 1000)
    .collect(engine='streaming')
)
t = df['t'].to_numpy()

def span(col):
    ts = df.filter(pl.col(col))['t']
    return ts.min(), ts.max()

f0, f1 = span(FREIGABE)
p0 = df.filter(pl.col('t') >= f0)[POS][0]
p1 = df.filter(pl.col('t') >= f1)[POS][0]

plt.rcParams.update({'font.size': 7, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig, axes = plt.subplots(3, 1, figsize=(6.6, 3.9), sharex=True, constrained_layout=True)

for ax in axes:
    ax.axvspan(f0, f1, color='#e9e9e6', zorder=0)
    ax.axvline(0, color=MUTED, ls=':', lw=0.8)
    ax.grid(color='#e3e3e3', lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)

ax = axes[0]
ax.step(t, df[POS], where='post', color=BLUE, lw=1.2)
ax.plot([f0, f1], [p0, p1], 'o', color=BLUE, ms=4, mec='white', mew=1)
ax.annotate(f'{p0:.0f} mm', (f0, p0), xytext=(6, -11), textcoords='offset points', color=MUTED)
ax.annotate(f'{p1:.0f} mm', (f1, p1), xytext=(6, -11), textcoords='offset points', color=MUTED)
ax.text((f0 + f1) / 2, 1.0, 'Reglerfreigabe', transform=ax.get_xaxis_transform(),
        ha='center', va='top', color=MUTED)
ax.set_title('C-Wagen – Istposition [mm]  (Punkte: Reglerfreigabe ein / aus)', loc='left', color=INK)

ax = axes[1]
ax.step(t, df[VEL], where='post', color=BLUE, lw=1.2)
ax.set_title('C-Wagen – Geschwindigkeit Ist [mm/s]', loc='left', color=INK)

ax = axes[2]
ax.step(t, df[I_OUT] / 1000, where='post', color=ORANGE, lw=1.2, label='Reglerausgang')
ax.step(t, df[I_IST] / 1000, where='post', color=BLUE, lw=1.2, label='Istwert (HKS Zuführung)')
ax.step(t, df[I_SOLL] / 1000, where='post', color=INK, lw=1, ls='--', label='Sollwert skaliert')
ax.legend(loc='center left', frameon=False)
ax.set_title('Stromregelung – Schweißstrom [kA]', loc='left', color=INK)
ax.set_xlabel('Zeit ab „Schweißen EIN“ [s]', color=MUTED)
ax.set_xlim(t.min(), t.max())

fig.savefig('Images/seq2_cwagen_stromregelung.png', dpi=300)
