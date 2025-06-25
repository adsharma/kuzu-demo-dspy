"""
Plot test results for different LLMs' ability to write the required quality of Cypher to
answer the user's questions.
"""

import matplotlib.pyplot as plt
import numpy as np

# Test results data
vanilla_results = {
    "openai/gpt-4o": "FFFFFFFFFF",
    "openai/gpt-4.1": "....FFFFFF",
    "google/gemini-2.0-flash": ".F..FFF.FF",
    "google/gemini-2.5-flash": ".FF.FFF.FF",
    "microsoft/phi4": "FFFFFFF.FF",
    "qwen/qwen3-30b-a3b": "F.F.FFF.FF",
    "mistralai/mistral-medium": "FFFFFFFFFF",
}

router_results = {
    "openai/gpt-4o": "FFFF......",
    "openai/gpt-4.1": "..........",
    "google/gemini-2.0-flash": "..........",
    "google/gemini-2.5-flash": "..F.......",
    "microsoft/phi4": "FFFF......",
    "qwen/qwen3-30b-a3b": "..F..FF..F",
    "mistralai/mistral-medium": "F.FF..F...",
}


def parse_results(results_dict):
    """Convert string results to binary matrix"""
    models = list(results_dict.keys())
    num_tests = len(list(results_dict.values())[0])

    matrix = np.zeros((len(models), num_tests))

    for i, (model, result_str) in enumerate(results_dict.items()):
        for j, char in enumerate(result_str):
            matrix[i, j] = 1 if char == "." else 0  # 1 for pass (.), 0 for fail (F)

    return matrix, models


def create_heatmap(matrix, models, title, filename):
    """Create and save heatmap"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create custom colormap: red for fail (0), green for pass (1)
    colors = ["#FF6961", "#77DD77"]  # Red for fail, green for pass
    cmap = plt.cm.colors.ListedColormap(colors)

    # Create heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Set labels
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([f"Q{i+1}" for i in range(matrix.shape[1])])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    # Add title
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75])
    cbar.set_ticklabels(["Fail", "Pass"])

    # Add grid between cells
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(True, which="minor", color="white", linewidth=0.5)

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    # plt.show()


def main():
    # Parse results
    vanilla_matrix, vanilla_models = parse_results(vanilla_results)
    router_matrix, router_models = parse_results(router_results)

    # Create heatmaps
    create_heatmap(
        vanilla_matrix,
        vanilla_models,
        "Vanilla Graph RAG",
        "vanilla_graph_rag_heatmap.png",
    )

    create_heatmap(
        router_matrix,
        router_models,
        "Router Agent Graph RAG",
        "router_agent_graph_rag_heatmap.png",
    )

    # Print summary statistics
    print("Summary Statistics:")
    print("\nVanilla Graph RAG:")
    for model, result_str in vanilla_results.items():
        passes = result_str.count(".")
        fails = result_str.count("F")
        print(
            f"  {model}: {passes}/{len(result_str)} tests passed ({passes/len(result_str)*100:.1f}%)"
        )

    print("\nRouter Agent Graph RAG:")
    for model, result_str in router_results.items():
        passes = result_str.count(".")
        fails = result_str.count("F")
        print(
            f"  {model}: {passes}/{len(result_str)} tests passed ({passes/len(result_str)*100:.1f}%)"
        )


if __name__ == "__main__":
    main()
