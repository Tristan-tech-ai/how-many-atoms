"""House figure style for paper 1 (paper_onelaw/FIGURE_GUIDE.md): IEEE one-column width 3.5 in, 9 pt text (8 pt ticks and
legend), STIX serif to match Times in IEEEtran, Okabe-Ito colours, embedded TrueType fonts, no gridlines, left/bottom spines."""
import matplotlib as mpl

OK = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
WIDTH = 3.5


def apply():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "legend.frameon": False,
    })
