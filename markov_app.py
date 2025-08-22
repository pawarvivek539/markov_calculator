import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Arc
import matplotlib as mpl
import streamlit as st
from background import show_background

# ---------- Drawing helpers ----------
def draw_self_loop(ax, x, y, prob, cmap, norm, radius=0.15):
    color = cmap(norm(prob))
    loop = Arc((x, y), width=2*radius, height=2*radius, angle=0,
               theta1=40, theta2=320, color=color, linewidth=2 + 5*prob)
    ax.add_patch(loop)
    arrow_x = x + radius * np.cos(np.deg2rad(320))
    arrow_y = y + radius * np.sin(np.deg2rad(320))
    arrow_dx = 0.01 * np.cos(np.deg2rad(320))
    arrow_dy = 0.01 * np.sin(np.deg2rad(320))
    arrow = FancyArrowPatch((arrow_x, arrow_y),
                           (arrow_x + arrow_dx, arrow_y + arrow_dy),
                           arrowstyle='-|>', mutation_scale=15,
                           color=color, linewidth=2 + 5*prob)
    ax.add_patch(arrow)
    ax.text(x, y + radius + 0.05, f"{prob:.2f}", ha='center', va='bottom',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="white", alpha=0.8))

def draw_unidirectional_transition(ax, x1, y1, x2, y2, prob, cmap, norm):
    color = cmap(norm(prob))
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    dx, dy = dx/dist, dy/dist
    start_x, start_y = x1 + dx * 0.1, y1 + dy * 0.1
    end_x, end_y = x2 - dx * 0.1, y2 - dy * 0.1
    arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                           arrowstyle='-|>', mutation_scale=20,
                           color=color, linewidth=2 + 5*prob)
    ax.add_patch(arrow)
    mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
    perp_dx, perp_dy = -dy, dx
    ax.text(mid_x + perp_dx * 0.1, mid_y + perp_dy * 0.1,
            f"{prob:.2f}", ha='center', va='center',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="white", alpha=0.8))

def draw_bidirectional_transition(ax, x1, y1, x2, y2, prob1, prob2, cmap, norm):
    color1 = cmap(norm(prob1))
    color2 = cmap(norm(prob2))
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    dx, dy = dx/dist, dy/dist
    perp_dx, perp_dy = -dy, dx
    start1_x, start1_y = x1 + dx * 0.1, y1 + dy * 0.1
    end1_x, end1_y = x2 - dx * 0.1, y2 - dy * 0.1
    curve_strength = 0.3
    control1_x = (start1_x + end1_x)/2 + perp_dx * curve_strength
    control1_y = (start1_y + end1_y)/2 + perp_dy * curve_strength
    control2_x = (start1_x + end1_x)/2 - perp_dx * curve_strength
    control2_y = (start1_y + end1_y)/2 - perp_dy * curve_strength
    path1 = mpl.path.Path([(start1_x, start1_y),
                          (control1_x, control1_y),
                          (end1_x, end1_y)],
                         [mpl.path.Path.MOVETO, mpl.path.Path.CURVE3, mpl.path.Path.CURVE3])
    arrow1 = FancyArrowPatch(path=path1, arrowstyle='-|>', mutation_scale=20,
                            color=color1, linewidth=2 + 5*prob1)
    ax.add_patch(arrow1)
    path2 = mpl.path.Path([(x2 - dx*0.1, y2 - dy*0.1),
                          (control2_x, control2_y),
                          (x1 + dx*0.1, y1 + dy*0.1)],
                         [mpl.path.Path.MOVETO, mpl.path.Path.CURVE3, mpl.path.Path.CURVE3])
    arrow2 = FancyArrowPatch(path=path2, arrowstyle='-|>', mutation_scale=20,
                            color=color2, linewidth=2 + 5*prob2)
    ax.add_patch(arrow2)
    ax.text(control1_x + perp_dx*0.15, control1_y + perp_dy*0.15,
            f"{prob1:.2f}", ha='center', va='center',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="white", alpha=0.8))
    ax.text(control2_x - perp_dx*0.15, control2_y - perp_dy*0.15,
            f"{prob2:.2f}", ha='center', va='center',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="white", alpha=0.8))

# ---------- Streamlit App ----------
def main():
    st.title("🔗 N-State Markov Chain Calculator")

    n = st.sidebar.number_input("Number of states (2–10)", min_value=2, max_value=10, value=2)

    st.sidebar.subheader("Transition Matrix")
    P = []
    for i in range(n):
        row = []
        for j in range(n):
            val = st.sidebar.number_input(f"P{i+1}{j+1} (S{i+1}→S{j+1})",
                                          min_value=0.0, max_value=1.0, value=0.0, step=0.05)
            row.append(val)
        P.append(row)
    P = np.array(P)

    st.sidebar.subheader("Initial State Probabilities")
    initial_state = []
    for i in range(n):
        val = st.sidebar.number_input(f"P0{i+1} (Initial in S{i+1})",
                                      min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        initial_state.append(val)
    initial_state = np.array(initial_state)

    steps = st.sidebar.number_input("Number of steps", min_value=1, max_value=50, value=5)

    if st.sidebar.button("Calculate"):
        current_state = initial_state
        history = [current_state]
        for _ in range(steps):
            current_state = np.dot(current_state, P)
            history.append(current_state)
        history = np.array(history)

        st.subheader("📊 State Probabilities Over Time")
        st.dataframe({f"Step {i}": history[i] for i in range(steps+1)})

                # --- Transition Diagram ---
        fig_size = 6 if n <= 4 else 8  # bigger figure for more states
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        ax.set_aspect('equal')
        ax.axis('off')

        # arrange nodes on a circle of radius 2 (instead of 1)
        radius = 2
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        positions = [(radius*np.cos(a), radius*np.sin(a)) for a in angles]

        node_radius = 0.2
        for i, (x, y) in enumerate(positions):
            circle = Circle((x, y), node_radius, facecolor='lightblue',
                            edgecolor='black', linewidth=2, alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y, f"S{i+1}", ha='center', va='center',
                    fontsize=14, fontweight='bold')

        cmap = plt.colormaps.get_cmap('Blues')
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        drawn = set()
        for i in range(n):
            for j in range(n):
                if P[i, j] > 0 and (i, j) not in drawn:
                    if i == j:
                        draw_self_loop(ax, *positions[i], P[i, j], cmap, norm)
                        drawn.add((i, j))
                    elif P[j, i] > 0:
                        draw_bidirectional_transition(ax, *positions[i], *positions[j],
                                                      P[i, j], P[j, i], cmap, norm)
                        drawn.add((i, j)); drawn.add((j, i))
                    else:
                        draw_unidirectional_transition(ax, *positions[i], *positions[j],
                                                       P[i, j], cmap, norm)
                        drawn.add((i, j))

        # expand limits so nothing is cropped
        ax.set_xlim(-radius-1, radius+1)
        ax.set_ylim(-radius-1, radius+1)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label('Transition Probability', fontsize=12)

        plt.tight_layout()
        st.pyplot(fig)
    # --- show historical context at the end ---
    st.markdown("---")
    if st.checkbox("📜 Show Historical Background (Markov’s Original Experiment)"):
        show_background()    

if __name__ == "__main__":
    main()
