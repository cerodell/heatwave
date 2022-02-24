lats_study = np.arange(45, 52 + 0.1, 0.1)
lons_study = np.arange(-123, -119 + 0.1, 0.1)
## mesh for ploting...could do without
lons_study, lats_study = np.meshgrid(lons_study, lats_study)

color = "black"
ax.plot(lons_study[0], lats_study[0], color=color, linewidth=2, zorder=8, alpha=1)
ax.plot(lons_study[-1].T, lats_study[-1].T, color=color, linewidth=2, zorder=8, alpha=1)
ax.plot(lons_study[:, 0], lats_study[:, 0], color=color, linewidth=2, zorder=8, alpha=1)
ax.plot(
    lons_study[:, -1].T,
    lats_study[:, -1].T,
    color=color,
    linewidth=2,
    zorder=8,
    alpha=1,
    label="Study Area",
)
