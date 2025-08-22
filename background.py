import streamlit as st
import numpy as np
import pandas as pd

def show_background():
    # --- History & Theory ---
    st.header("📜 Markov’s Original Experiment and Theory")

    st.subheader("1. Markov’s Theorem")
    st.markdown(
        """
        Origin: Developed by Andrey Markov, expanding probability theory to dependent events.

        **Nekrasov vs. Markov**: Pavel Nekrasov (mathematician/theologian) argued the law of large numbers required independent events only linking this to philosophical arguments about free will and the independence of acts.  
        **Markov disagreed**, showing the law applies even when events are dependent.

        **Key Point:** Markov chains model systems where the next event depends only on the *current state*, not complete independence.
        """
    )

    # --- Markov’s Proof ---
    st.subheader("2. Markov’s Proof Using *Eugene Onegin*")
    st.markdown(
        """
        Markov analyzed **20,000 letters** of Pushkin’s poem *Eugene Onegin*, classifying letters as vowels (V) or consonants (C).  

        Transition counts:

        - V→V = 1,104  
        - V→C = 7,534  
        - C→V = 7,535  
        - C→C = 3,827  

        Totals:  
        - Vowels = 8,638 (43%)  
        - Consonants = 11,362 (57%)  
        """
    )

    # --- Transition Matrix ---
    st.subheader("3. Transition Matrix")
    st.latex(r"""
    P = \begin{bmatrix}
    P(V \to V) & P(V \to C) \\
    P(C \to V) & P(C \to C)
    \end{bmatrix}
    =
    \begin{bmatrix}
    0.128 & 0.872 \\
    0.664 & 0.337
    \end{bmatrix}
    """)

    st.markdown(
        """
        - P(V→V): Probability next letter is a vowel given current is a vowel  
        - P(V→C): Probability next is a consonant given current is a vowel  
        - P(C→V): Probability next is a vowel given current is a consonant  
        - P(C→C): Probability next is consonant given current is consonant  

        From a vowel: **12.8% chance vowel, 87.2% consonant**  
        From a consonant: **66.4% chance vowel, 33.7% consonant**  
        """
    )

    # --- Calculation Demo ---
    st.subheader("4. Example: Evolving State Probabilities")

    st.markdown("The general formula for state probabilities in a Markov chain is:")

    st.latex(r"u_1 = u_0 \cdot P")

    st.markdown("After \(n\) steps:")

    st.latex(r"u_n = u_0 \cdot P^n")

    st.markdown("where:")

    st.latex(r"u_0 \;=\; \text{initial distribution}")
    st.latex(r"P \;=\; \text{transition matrix}")
    st.latex(r"u_n \;=\; \text{distribution after } n \text{ steps}")

    st.markdown("👉 Assuming the initial state distribution is:")

    # Transition matrix from Markov's experiment (V,C order)
    transition_matrix = np.array([
        [0.128, 0.872],
        [0.664, 0.337]
    ])

    # Initial probabilities (V, C)
    p0 = np.array([0.30, 0.70])

    # First step calculation
    p1 = p0 @ transition_matrix

    # Show u0, P, and the multiplication u1 = u0 · P
    st.latex(r"u_0 = \begin{bmatrix} 0.30 & 0.70 \end{bmatrix}")
    st.latex(r"""
    P =
    \begin{bmatrix}
    0.128 & 0.872 \\
    0.664 & 0.337
    \end{bmatrix}
    """)
    st.latex(r"u_1 = u_0 \cdot P")
    st.latex(r"""
    u_1 =
    \begin{bmatrix} 0.30 & 0.70 \end{bmatrix}
    \cdot
    \begin{bmatrix}
    0.128 & 0.872 \\
    0.664 & 0.337
    \end{bmatrix}
    """)

    # Safely render the computed u1 as LaTeX (note the doubled braces)
    latex_result = (
        r"u_1 = \begin{{bmatrix}} {0:.4f} & {1:.4f} \end{{bmatrix}}"
    ).format(p1[0], p1[1])
    st.latex(latex_result)

    st.markdown(
        r"""
    🔁 To get the **next step**, multiply the new distribution by the same transition matrix again:
        """
    )
    st.latex(r"u_2 = u_1 \cdot P \quad then \quad u_{3} = u_2 \cdot P")
    st.latex(r"till\quad u_{n+1} = u_n \cdot P")


    st.markdown(
        r"""
    Repeating this process eventually converges to the **stationary distribution** (here ≈ \( (0.43, 0.57) \)).
        """
    )

    # Multiple steps simulation and table
    steps = 15
    probs = [p0]
    for _ in range(steps):
        p_next = probs[-1] @ transition_matrix
        probs.append(p_next)

    df = pd.DataFrame(probs, columns=["Vowel (V)", "Consonant (C)"])
    df.index.name = "Step"

    st.markdown("**Probabilities across steps:**")
    st.dataframe(df.style.format("{:.4f}"))

    # --- Conclusion ---
    st.markdown(
        """
        ✅ **Conclusion:**  
        Markov demonstrated that regardless of the **initial probabilities**,  
        and even when events are **dependent** (like letters following each other),  
        the distribution always **converges** to a stable long-run proportion.  

        In this case, the chain converges to approximately:  
        - Vowels ≈ **0.43**  
        - Consonants ≈ **0.57**  

        🔑 This matches what Markov observed in *Eugene Onegin* after analyzing  
        20,000 letters, proving that the **Law of Large Numbers** holds even  
        for dependent events.  

        Moreover, the process is **memoryless** ,only the current state matters,  
        not the entire history of past states.  
        """
    )

    # --- Applications ---
    st.subheader("5. Modern Applications")
    st.markdown(
        """
        Markov chains are now widely used in:
        - **Natural Language Processing (NLP):** predictive text, speech recognition  
        - **Google PageRank:** web navigation modeled as a random walk  
        - **Finance:** stock price models  
        - **Bioinformatics:** DNA sequence analysis  
        - **AI & Reinforcement Learning:** state transitions in decision-making  
        """
    )
