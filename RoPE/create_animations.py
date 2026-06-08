import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

output_dir = os.path.dirname(os.path.abspath(__file__))
d = 768
half_d = d // 2
base = 10000
alphas = base ** (-2 * torch.arange(half_d, dtype=torch.float) / d).numpy()

# ========== Animation 1: Group orbits with live tracing ==========
print('Creating Animation 1: Group orbits ...')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
mu_vec = np.array([1.0, 0.5])
total_frames = 200
theta_max = 6 * np.pi
thetas = np.linspace(0, theta_max, total_frames)

# Precompute orbits
decay_rate = 0.08
x_circ = mu_vec[0]*np.cos(thetas) - mu_vec[1]*np.sin(thetas)
y_circ = mu_vec[0]*np.sin(thetas) + mu_vec[1]*np.cos(thetas)
decay = np.exp(-decay_rate * thetas)
x_spiral = decay * x_circ
y_spiral = decay * y_circ

lines_circ, = axes[0].plot([], [], linewidth=0.8, color='#1565C0', alpha=0.7)
point_circ, = axes[0].plot([], [], 'ro', markersize=8)
fixed_q = axes[0].plot([0, mu_vec[0]], [0, mu_vec[1]], 'g--', linewidth=1.5, alpha=0.7, label='mu_q (fixed)')[0]
axes[0].set_title('RoPE: Circular Orbit', fontsize=12)
axes[0].set_aspect('equal'); axes[0].axhline(y=0, color='gray', ls='--', alpha=0.3)
axes[0].axvline(x=0, color='gray', ls='--', alpha=0.3)
axes[0].set_xlim(-1.5, 1.5); axes[0].set_ylim(-1.5, 1.5)
axes[0].legend(loc='upper right', fontsize=8)
inner_prod_text_circ = axes[0].text(0.05, 0.95, '', transform=axes[0].transAxes,
                                     va='top', fontsize=9,
                                     bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

# Annotate angle on the circular plot: we'll show dot product
angle_arrow_circ = axes[0].annotate('', xy=(0, 0), xytext=(0, 0),
                                     arrowprops=dict(arrowstyle='->', color='purple', lw=2, alpha=0))

lines_spi, = axes[1].plot([], [], linewidth=0.8, color='#C62828', alpha=0.7)
point_spi, = axes[1].plot([], [], 'ro', markersize=8)
axes[1].plot(0, 0, 'kx', markersize=8, label='origin (attractor)')
axes[1].plot([0, mu_vec[0]], [0, mu_vec[1]], 'g--', linewidth=1.5, alpha=0.7, label='mu_q (fixed)')
axes[1].set_title('Damped RoPE: Spiral Orbit', fontsize=12)
axes[1].set_aspect('equal'); axes[1].axhline(y=0, color='gray', ls='--', alpha=0.3)
axes[1].axvline(x=0, color='gray', ls='--', alpha=0.3)
axes[1].set_xlim(-1.5, 1.5); axes[1].set_ylim(-1.5, 1.5)
axes[1].legend(loc='upper right', fontsize=8)
inner_prod_text_spi = axes[1].text(0.05, 0.95, '', transform=axes[1].transAxes,
                                    va='top', fontsize=9,
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

# Zero mean: show random cloud + one vector being rotated
np.random.seed(42)
cloud_x = np.random.randn(100)*0.3
cloud_y = np.random.randn(100)*0.3
axes[2].scatter(cloud_x, cloud_y, s=4, alpha=0.3, color='#78909C', label='q,k samples')
zero_point, = axes[2].plot([], [], 'ro', markersize=6, label='rotated sample')
zero_trace, = axes[2].plot([], [], linewidth=0.6, color='#C62828', alpha=0.5)
axes[2].set_title('mu=0: No Coherent Orbit', fontsize=12)
axes[2].set_aspect('equal'); axes[2].axhline(y=0, color='gray', ls='--', alpha=0.3)
axes[2].axvline(x=0, color='gray', ls='--', alpha=0.3)
axes[2].set_xlim(-1.5, 1.5); axes[2].set_ylim(-1.5, 1.5)
axes[2].legend(loc='upper right', fontsize=8)

plt.suptitle('Group Orbits in 2D Subspace (animated)', fontsize=14, y=1.02)
plt.tight_layout()

def init():
    lines_circ.set_data([], []); point_circ.set_data([], [])
    lines_spi.set_data([], []); point_spi.set_data([], [])
    zero_point.set_data([], []); zero_trace.set_data([], [])
    inner_prod_text_circ.set_text(''); inner_prod_text_spi.set_text('')
    return lines_circ, point_circ, lines_spi, point_spi, zero_point, zero_trace

def update(frame):
    n = frame + 1
    # Circular orbit
    lines_circ.set_data(x_circ[:n], y_circ[:n])
    point_circ.set_data([x_circ[frame]], [y_circ[frame]])
    # dot product between mu_q and rotated mu_k
    dot_circ = np.dot(mu_vec, [x_circ[frame], y_circ[frame]])
    inner_prod_text_circ.set_text(f'<mu_q, mu_k(rotated)> = {dot_circ:.3f}')

    # Spiral orbit
    lines_spi.set_data(x_spiral[:n], y_spiral[:n])
    point_spi.set_data([x_spiral[frame]], [y_spiral[frame]])
    dot_spi = np.dot(mu_vec, [x_spiral[frame], y_spiral[frame]])
    inner_prod_text_spi.set_text(f'<mu_q, mu_k(rotated)> = {dot_spi:.3f}')

    # Zero-mean: pick a sample from cloud, rotate it
    # Use a small vector from the cloud
    if frame < 50:
        idx = 0
    else:
        idx = min(frame - 50, 99)
    v = np.array([cloud_x[idx], cloud_y[idx]])
    theta = thetas[min(frame, len(thetas)-1)]
    v_norm = np.linalg.norm(v)
    if v_norm > 0.01:
        v_rot_x = v[0]*np.cos(theta) - v[1]*np.sin(theta)
        v_rot_y = v[0]*np.sin(theta) + v[1]*np.cos(theta)
        trace_len = min(n, 40)
        t_start = max(0, frame - trace_len)
        trace_thetas = thetas[t_start:frame+1]
        trace_x = v[0]*np.cos(trace_thetas) - v[1]*np.sin(trace_thetas)
        trace_y = v[0]*np.sin(trace_thetas) + v[1]*np.cos(trace_thetas)
        zero_trace.set_data(trace_x, trace_y)
        zero_point.set_data([v_rot_x], [v_rot_y])

    return lines_circ, point_circ, lines_spi, point_spi, zero_point, zero_trace, inner_prod_text_circ, inner_prod_text_spi

ani = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=False, interval=30)
writer = PillowWriter(fps=25)
ani.save(os.path.join(output_dir, 'animation_orbit.gif'), writer=writer)
plt.close()
print('  -> animation_orbit.gif saved')


# ========== Animation 2: SNR sweep - continuous transition ==========
print('Creating Animation 2: SNR sweep ...')

seq_len = 2000
output_dim = 768
frequency = 10000
device = 'cpu'

position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
indices = torch.arange(0, output_dim // 2, dtype=torch.float)
indices = torch.pow(frequency, -2 * indices / output_dim)
embeddings = position_ids * indices
embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
embeddings = embeddings.unsqueeze(0)
embeddings = torch.reshape(embeddings, (1, seq_len, output_dim)).to(device)
cos_pos = embeddings[..., 1::2].repeat_interleave(2, dim=-1)
sin_pos = embeddings[..., ::2].repeat_interleave(2, dim=-1)

fig2, ax2 = plt.subplots(figsize=(10, 5))
line2, = ax2.plot([], [], linewidth=0.8, color='#1565C0')
snr_text = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, va='top', fontsize=12,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
ax2.set_xlim(0, seq_len); ax2.set_ylim(-0.6, 1.1)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
ax2.set_xlabel('Relative distance'); ax2.set_ylabel('Normalized score')
ax2.set_title('SNR Sweep:   mu=1, sigma varies', fontsize=13)
plt.tight_layout()

mu_val = 1.0
sigmas = np.linspace(0.1, 3.5, 200)

def update2(frame):
    sigma = sigmas[frame]
    snr = mu_val**2 / sigma**2
    torch.manual_seed(42)
    q = torch.normal(mean=mu_val, std=sigma, size=(1, seq_len, output_dim)).to(device)
    k = torch.normal(mean=mu_val, std=sigma, size=(1, seq_len, output_dim)).to(device)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], -1).reshape(q.shape)
    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], -1).reshape(k.shape)
    q_rope = q * cos_pos + q2 * sin_pos
    k_rope = k * cos_pos + k2 * sin_pos
    score_rope = torch.einsum('bmd,bnd->bmn', q_rope, k_rope)
    decay = torch.flip(score_rope[0][-1] / torch.max(score_rope[0][-1]), dims=[0]).cpu().numpy()
    line2.set_data(np.arange(seq_len), decay)
    snr_text.set_text(f'mu=1, sigma={sigma:.2f}   SNR={snr:.1f}')
    return line2, snr_text

ani2 = FuncAnimation(fig2, update2, frames=len(sigmas), blit=False, interval=40)
writer2 = PillowWriter(fps=20)
ani2.save(os.path.join(output_dir, 'animation_snr_sweep.gif'), writer=writer2)
plt.close()
print('  -> animation_snr_sweep.gif saved')


# ========== Animation 3: Multi-frequency interference buildup ==========
print('Creating Animation 3: Multi-frequency buildup ...')

fig3, (ax3_top, ax3_bot) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 1.5]})
taus = np.arange(0, 2000)
n_freqs_total = half_d
# subsample frequencies for visual clarity
freq_indices = np.linspace(0, n_freqs_total-1, 60, dtype=int)
alphas_sub = alphas[freq_indices]

all_components = np.array([np.cos(alpha * taus) for alpha in alphas_sub])
cumulative = np.cumsum(all_components, axis=0)

line_cumul, = ax3_bot.plot([], [], linewidth=0.8, color='#C62828')
line_highlight, = ax3_top.plot([], [], linewidth=0.6, color='#1565C0', alpha=0.8)
freq_text = ax3_top.text(0.05, 0.95, '', transform=ax3_top.transAxes, va='top', fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax3_bot.set_xlim(0, 2000); ax3_bot.set_ylim(-60, 80)
ax3_bot.axhline(y=0, color='gray', ls='--', alpha=0.3)
ax3_bot.set_xlabel('Relative distance tau')
ax3_bot.set_ylabel('Sum of cos(alpha_i * tau)')
ax3_bot.set_title('Cumulative sum: adding frequency components', fontsize=11)
ax3_top.set_xlim(0, 2000); ax3_top.set_ylim(-1.5, 1.5)
ax3_top.axhline(y=0, color='gray', ls='--', alpha=0.3)
ax3_top.set_ylabel('cos(alpha_i * tau)')
ax3_top.set_title('Single frequency component (highlighted)', fontsize=11)
plt.tight_layout()

def update3(frame):
    n = frame + 1
    ax3_bot.set_title(f'Cumulative sum: {n}/{len(alphas_sub)} frequency components', fontsize=11)
    line_cumul.set_data(taus, cumulative[n-1] / n)  # normalize by count
    freq_text.set_text(f'freq index = {freq_indices[min(n-1, len(freq_indices)-1)]}, alpha = {alphas_sub[min(n-1, len(freq_indices)-1)]:.6f}')
    # Show the latest added component
    line_highlight.set_data(taus, all_components[min(n-1, len(freq_indices)-1)])
    return line_cumul, line_highlight, freq_text

ani3 = FuncAnimation(fig3, update3, frames=min(60, len(alphas_sub)), blit=False, interval=150)
writer3 = PillowWriter(fps=5)
ani3.save(os.path.join(output_dir, 'animation_freq_buildup.gif'), writer=writer3)
plt.close()
print('  -> animation_freq_buildup.gif saved')


# ========== Animation 4: SNR effect on group orbit ==========
print('Creating Animation 4: SNR orbit visualization ...')

fig4, ax4 = plt.subplots(figsize=(7, 7))
ax4.set_xlim(-2.5, 2.5); ax4.set_ylim(-2.5, 2.5)
ax4.set_aspect('equal')
ax4.axhline(y=0, color='gray', ls='--', alpha=0.3)
ax4.axvline(x=0, color='gray', ls='--', alpha=0.3)
ax4.set_title('Orbit + noise: SNR changes\nRotating vector trajectory', fontsize=12)

orbit_line, = ax4.plot([], [], linewidth=1.0, color='#1565C0', alpha=0.5)
noise_scatter = ax4.scatter([], [], s=3, alpha=0.2, color='#78909C')
point, = ax4.plot([], [], 'ro', markersize=7)
snr_text4 = ax4.text(0.05, 0.95, '', transform=ax4.transAxes, va='top', fontsize=11,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
plt.tight_layout()

theta4 = np.linspace(0, 6*np.pi, 400)
x_clean = np.cos(theta4)
y_clean = np.sin(theta4)
noise_levels = np.linspace(0.01, 1.5, 150)

def update4(frame):
    sigma = noise_levels[frame]
    # signal magnitude
    mu_norm = 1.0
    n_pts = 400
    # add noise to the orbit
    np.random.seed(42)
    noise_x = np.random.randn(n_pts) * sigma
    noise_y = np.random.randn(n_pts) * sigma
    x_noisy = x_clean + noise_x
    y_noisy = y_clean + noise_y
    orbit_line.set_data(x_noisy, y_noisy)
    # scatter: random samples from noisy distribution
    scatter_pts = 200
    np.random.seed(42)
    sx = mu_norm * np.cos(np.random.rand(scatter_pts)*2*np.pi) + np.random.randn(scatter_pts)*sigma
    sy = mu_norm * np.sin(np.random.rand(scatter_pts)*2*np.pi) + np.random.randn(scatter_pts)*sigma
    noise_scatter.set_offsets(np.c_[sx, sy])
    # current point
    idx = frame % n_pts
    point.set_data([x_noisy[idx]], [y_noisy[idx]])
    snr = mu_norm**2 / sigma**2 if sigma > 0 else 999
    snr_text4.set_text(f'sigma = {sigma:.2f}   SNR = {snr:.1f}')
    return orbit_line, point, snr_text4, noise_scatter

ani4 = FuncAnimation(fig4, update4, frames=len(noise_levels), blit=False, interval=40)
writer4 = PillowWriter(fps=20)
ani4.save(os.path.join(output_dir, 'animation_snr_orbit.gif'), writer=writer4)
plt.close()
print('  -> animation_snr_orbit.gif saved')

print('\nAll animations created successfully!')
