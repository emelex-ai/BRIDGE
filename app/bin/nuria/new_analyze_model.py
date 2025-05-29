import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from bridge.domain.dataset import BridgeDataset
from bridge.domain.model import Model
import json
from scipy.spatial.distance import hamming
from scipy.stats import entropy
import warnings

warnings.filterwarnings("ignore")


def load_model_and_dataset(epoch_num: int):
    """Load the trained model and dataset."""
    print("Loading model and dataset...")
    chkpt = torch.load(
        f"model_artifacts/nuria_experiments/2025_05_27_run_001/model_epoch_{epoch_num}.pth",
        weights_only=False,
    )
    chkpt["dataset_config"].dataset_filepath = (
        "data/nuria_pretraining/p_p_bilingual_learner.csv"
    )
    dataset = BridgeDataset(chkpt["dataset_config"], None)
    model = Model(chkpt["model_config"], dataset)
    model.load_state_dict(
        chkpt["model_state_dict"]
    )  # CRITICAL: Load the trained weights!
    model.eval()
    print(f"Model loaded. Dataset contains {len(dataset)} words.")
    return model, dataset


def phoneme_tokens_to_string(tokens):
    """Convert phoneme token list to readable string."""
    return " ".join([str(t.tolist()) for t in tokens])


def features_to_binary_matrix(feature_list, n_features=35):
    """Convert list of feature tensors to binary matrix."""
    matrix = []
    for features in feature_list:
        if isinstance(features, torch.Tensor):
            matrix.append(features.cpu().numpy())
        else:
            # Convert feature indices to binary vector
            binary_vec = np.zeros(n_features)
            for idx in features:
                if idx < n_features:
                    binary_vec[idx] = 1
            matrix.append(binary_vec)
    return np.array(matrix)


def calculate_edit_distance(seq1, seq2):
    """Calculate edit distance between two sequences."""
    # Convert to comparable format
    str1 = phoneme_tokens_to_string(seq1)
    str2 = phoneme_tokens_to_string(seq2)

    # Simple edit distance
    if len(str1) == 0:
        return len(str2)
    if len(str2) == 0:
        return len(str1)

    # Create matrix
    matrix = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]

    for i in range(len(str1) + 1):
        matrix[i][0] = i
    for j in range(len(str2) + 1):
        matrix[0][j] = j

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                matrix[i][j] = 1 + min(
                    matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1]
                )

    return matrix[len(str1)][len(str2)]


def calculate_confidence_metrics(probabilities):
    """Calculate various confidence metrics from probability distributions."""
    if not probabilities:
        return {
            "avg_max_prob": 0.0,
            "min_max_prob": 0.0,
            "max_max_prob": 0.0,
            "avg_entropy": 0.0,
            "min_entropy": 0.0,
            "max_entropy": 0.0,
        }

    max_probs = []
    entropies = []

    for prob_dist in probabilities:
        if isinstance(prob_dist, torch.Tensor):
            prob_np = prob_dist.cpu().numpy()
            max_probs.append(np.max(prob_np))
            # Calculate entropy, handling zero probabilities
            prob_np_safe = np.clip(prob_np, 1e-10, 1.0)
            entropies.append(entropy(prob_np_safe))

    return {
        "avg_max_prob": np.mean(max_probs) if max_probs else 0.0,
        "min_max_prob": np.min(max_probs) if max_probs else 0.0,
        "max_max_prob": np.max(max_probs) if max_probs else 0.0,
        "avg_entropy": np.mean(entropies) if entropies else 0.0,
        "min_entropy": np.min(entropies) if entropies else 0.0,
        "max_entropy": np.max(entropies) if entropies else 0.0,
    }


def calculate_feature_level_metrics(target_matrix, generated_matrix):
    """Calculate detailed feature-level accuracy metrics."""
    n_features = target_matrix.shape[1] if target_matrix.size > 0 else 35

    # OPTIMIZATION: Skip padding if matrices are same size
    if target_matrix.shape[0] == generated_matrix.shape[0]:
        target_padded = target_matrix
        generated_padded = generated_matrix
    else:
        # Only pad when necessary
        max_len = max(target_matrix.shape[0], generated_matrix.shape[0])
        target_padded = np.full((max_len, n_features), 2)
        generated_padded = np.full((max_len, n_features), 2)
        target_padded[: target_matrix.shape[0], :] = target_matrix
        generated_padded[: generated_matrix.shape[0], :] = generated_matrix

    # OPTIMIZATION: Vectorized calculation for all features at once
    feature_metrics = {}

    # Create mask for all valid positions (not padding)
    valid_mask = target_padded != 2  # Shape: (seq_len, n_features)

    # OPTIMIZATION: Only calculate for features that have valid positions
    features_with_data = np.any(valid_mask, axis=0)  # Shape: (n_features,)

    for f in range(n_features):
        if features_with_data[f]:
            # OPTIMIZATION: Use boolean indexing directly
            target_valid = target_padded[:, f][valid_mask[:, f]]
            generated_valid = generated_padded[:, f][valid_mask[:, f]]

            # OPTIMIZATION: Vectorized equality check
            accuracy = np.mean(target_valid == generated_valid)

            # OPTIMIZATION: Simplified precision/recall calculation
            if len(target_valid) > 0:
                true_positives = np.sum((target_valid == 1) & (generated_valid == 1))
                false_positives = np.sum((target_valid == 0) & (generated_valid == 1))
                false_negatives = np.sum((target_valid == 1) & (generated_valid == 0))

                precision = (
                    true_positives / (true_positives + false_positives)
                    if (true_positives + false_positives) > 0
                    else 0.0
                )
                recall = (
                    true_positives / (true_positives + false_negatives)
                    if (true_positives + false_negatives) > 0
                    else 0.0
                )
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
            else:
                precision = recall = f1 = 0.0

            feature_metrics[f"feature_{f}_accuracy"] = accuracy
            feature_metrics[f"feature_{f}_precision"] = precision
            feature_metrics[f"feature_{f}_recall"] = recall
            feature_metrics[f"feature_{f}_f1"] = f1
        else:
            # OPTIMIZATION: Fast assignment for features with no data
            feature_metrics[f"feature_{f}_accuracy"] = np.nan
            feature_metrics[f"feature_{f}_precision"] = np.nan
            feature_metrics[f"feature_{f}_recall"] = np.nan
            feature_metrics[f"feature_{f}_f1"] = np.nan

    return feature_metrics


def evaluate_word_from_batch_results(
    word_idx,
    word_str,
    target_tokens,
    generated_tokens,
    target_features,
    generated_features,
    probabilities,
    global_encoding,
    dataset,
):
    """Process individual word results from batch generation."""

    # Convert generated features to matrix format
    generated_matrix = features_to_binary_matrix(generated_features)

    # Basic metrics
    word_exact_match = len(target_tokens) == len(generated_tokens) and all(
        torch.equal(t, g) for t, g in zip(target_tokens, generated_tokens)
    )
    sequence_length_match = len(target_tokens) == len(generated_tokens)
    edit_distance = calculate_edit_distance(target_tokens, generated_tokens)

    # Feature-level analysis
    overall_feature_accuracy = 0.0
    if target_features.size > 0 and generated_matrix.size > 0:
        min_len = min(target_features.shape[0], generated_matrix.shape[0])
        matches = 0
        total = 0
        for i in range(min_len):
            for j in range(target_features.shape[1]):
                if target_features[i, j] != 2:  # Not padding
                    total += 1
                    if target_features[i, j] == generated_matrix[i, j]:
                        matches += 1
        overall_feature_accuracy = matches / total if total > 0 else 0.0

    # Phoneme-level analysis
    phoneme_accuracies = []
    for i in range(min(len(target_tokens), len(generated_tokens))):
        # Compare the feature indices for this phoneme
        target_phoneme = set(target_tokens[i].tolist())
        generated_phoneme = set(generated_tokens[i].tolist())
        phoneme_exact_match = target_phoneme == generated_phoneme
        phoneme_accuracies.append(float(phoneme_exact_match))

    overall_phoneme_accuracy = (
        np.mean(phoneme_accuracies) if phoneme_accuracies else 0.0
    )

    # Confidence metrics
    confidence_metrics = calculate_confidence_metrics(probabilities)

    # Feature-level detailed metrics
    feature_level_metrics = {}
    # Turn off for now. This is computationally expensive and not needed yet
    # feature_level_metrics = calculate_feature_level_metrics(
    #    target_features, generated_matrix
    # )

    # Compile all results
    result = {
        "word_index": word_idx,
        "word_text": word_str,
        "language": (
            dataset.language_map.get(word_str.lower())
            if dataset.language_map
            else "unknown"
        ),
        "target_sequence": phoneme_tokens_to_string(target_tokens),
        "generated_sequence": phoneme_tokens_to_string(generated_tokens),
        "target_length": len(target_tokens),
        "generated_length": len(generated_tokens),
        "word_exact_match": word_exact_match,
        "sequence_length_match": sequence_length_match,
        "overall_feature_accuracy": overall_feature_accuracy,
        "overall_phoneme_accuracy": overall_phoneme_accuracy,
        "edit_distance": edit_distance,
        "global_encoding": json.dumps(global_encoding.tolist()),
        "target_features": json.dumps(target_features.tolist()),
        "generated_features": json.dumps(generated_matrix.tolist()),
        "probabilities": json.dumps([p.cpu().numpy().tolist() for p in probabilities]),
        **confidence_metrics,
        **feature_level_metrics,
    }

    # Add per-phoneme accuracy columns
    max_phonemes = 10  # Adjust based on your data
    for i in range(max_phonemes):
        result[f"phoneme_{i}_accuracy"] = (
            phoneme_accuracies[i] if i < len(phoneme_accuracies) else np.nan
        )

    return result, target_features, generated_matrix


def evaluate_batch_of_words(model, word_indices, dataset, batch_size=32):
    """Evaluate a batch of words using batched model inference."""

    # Get words for this batch
    words_in_batch = [dataset.words[idx] for idx in word_indices]

    try:
        # Get batch encoding - use list of words for batch processing
        batch_encoding = dataset[words_in_batch]

        # Single batched model inference call - this is the key optimization
        batch_generation = model.generate(
            batch_encoding, pathway="p2p", deterministic=True
        )

        # Process each word individually from batch results
        batch_results = []
        batch_targets = []
        batch_generated = []

        for i, word_idx in enumerate(word_indices):
            word_str = words_in_batch[i]

            # Extract individual results from batch
            target_tokens = batch_encoding.phonological.enc_input_ids[i]
            generated_tokens = batch_generation.phon_tokens[i]
            target_features = batch_encoding.phonological.targets[i].cpu().numpy()
            generated_features = batch_generation.phon_vecs[i]
            probabilities = batch_generation.phon_probs[i]
            global_encoding = batch_generation.global_encoding[i, 0, :].cpu().numpy()

            # Process this word's results
            result, target_feat, gen_feat = evaluate_word_from_batch_results(
                word_idx,
                word_str,
                target_tokens,
                generated_tokens,
                target_features,
                generated_features,
                probabilities,
                global_encoding,
                dataset,
            )

            if result is not None:
                batch_results.append(result)
                if target_feat is not None and gen_feat is not None:
                    batch_targets.append(target_feat)
                    batch_generated.append(gen_feat)

        return batch_results, batch_targets, batch_generated

    except Exception as e:
        print(f"Error evaluating batch {word_indices}: {e}")
        # Fallback to individual processing if batch fails
        return evaluate_batch_individually(model, word_indices, dataset)


def evaluate_batch_individually(model, word_indices, dataset):
    """Fallback method: evaluate words individually if batch processing fails."""
    batch_results = []
    batch_targets = []
    batch_generated = []

    for word_idx in word_indices:
        try:
            word_str = dataset.words[word_idx]
            encoding = dataset[word_idx]
            generation = model.generate(encoding, pathway="p2p", deterministic=True)

            target_tokens = encoding.phonological.enc_input_ids[0]
            generated_tokens = generation.phon_tokens[0]
            target_features = encoding.phonological.targets[0].cpu().numpy()
            generated_features = generation.phon_vecs[0]
            probabilities = generation.phon_probs[0]
            global_encoding = generation.global_encoding[0, 0, :].cpu().numpy()

            result, target_feat, gen_feat = evaluate_word_from_batch_results(
                word_idx,
                word_str,
                target_tokens,
                generated_tokens,
                target_features,
                generated_features,
                probabilities,
                global_encoding,
                dataset,
            )

            if result is not None:
                batch_results.append(result)
                if target_feat is not None and gen_feat is not None:
                    batch_targets.append(target_feat)
                    batch_generated.append(gen_feat)

        except Exception as e:
            print(f"Error evaluating word {word_idx}: {e}")
            continue

    return batch_results, batch_targets, batch_generated


def create_batches(dataset_size, batch_size):
    """Create batches of word indices."""
    batches = []
    for i in range(0, dataset_size, batch_size):
        end_idx = min(i + batch_size, dataset_size)
        batches.append(list(range(i, end_idx)))
    return batches


def convert_to_analysis_format(all_targets, all_generated):
    """Convert data to format expected by original analysis code."""
    if not all_targets or not all_generated:
        return None, None

    # Find max sequence length
    max_len = max(
        max(t.shape[0] for t in all_targets), max(g.shape[0] for g in all_generated)
    )
    n_features = all_targets[0].shape[1]
    n_words = len(all_targets)

    # Initialize tensors with padding value (2)
    phon_targets = torch.full((n_words, max_len, n_features), 2, dtype=torch.long)
    phon_preds = torch.full((n_words, max_len, n_features), 2, dtype=torch.long)

    # Fill in actual data
    for w in range(n_words):
        target_len = all_targets[w].shape[0]
        generated_len = all_generated[w].shape[0]

        phon_targets[w, :target_len, :] = torch.from_numpy(all_targets[w]).long()
        phon_preds[w, :generated_len, :] = torch.from_numpy(all_generated[w]).long()

    return phon_preds, phon_targets


def calculate_dataset_metrics(all_targets, all_generated):
    """Calculate dataset-level metrics using the original analysis logic."""
    # Convert to expected format
    phon_preds, phon_targets = convert_to_analysis_format(all_targets, all_generated)

    if phon_preds is None:
        return pd.DataFrame(
            {"metric": ["error"], "value": ["Could not calculate metrics"]}
        )

    # Apply original analysis logic
    phon_features_mask = phon_targets != 2
    masked_equalities = torch.eq(phon_preds, phon_targets) & phon_features_mask

    # FIXED: Use high precision and proper formatting
    feature_accuracy = (
        masked_equalities.sum().float() / phon_features_mask.sum().float()
    )
    phoneme_correct = masked_equalities.all(dim=2)
    valid_phonemes = ~(phon_targets == 2).all(dim=2)
    phoneme_accuracy = (
        phoneme_correct[valid_phonemes].sum().float() / valid_phonemes.sum().float()
    )
    word_correct = torch.all(phoneme_correct | ~valid_phonemes, dim=1)
    word_accuracy = word_correct.sum().float() / float(word_correct.size(0))

    # Additional dataset statistics
    results = []
    # FIXED: Show proper precision (6 decimal places)
    results.append(
        {
            "metric": "Overall Feature Accuracy",
            "value": f"{feature_accuracy.item():.6f}",
        }
    )
    results.append(
        {
            "metric": "Overall Phoneme Accuracy",
            "value": f"{phoneme_accuracy.item():.6f}",
        }
    )
    results.append(
        {"metric": "Overall Word Accuracy", "value": f"{word_accuracy.item():.6f}"}
    )
    results.append({"metric": "Total Words", "value": phon_targets.size(0)})
    results.append({"metric": "Words Correct", "value": word_correct.sum().item()})
    results.append(
        {"metric": "Total Features Evaluated", "value": phon_features_mask.sum().item()}
    )
    results.append(
        {"metric": "Features Correct", "value": masked_equalities.sum().item()}
    )
    results.append(
        {"metric": "Total Phonemes Evaluated", "value": valid_phonemes.sum().item()}
    )
    results.append(
        {
            "metric": "Phonemes Correct",
            "value": phoneme_correct[valid_phonemes].sum().item(),
        }
    )

    # Per-feature analysis - FIXED: Use proper precision
    for f in range(phon_targets.size(2)):
        feature_mask = phon_features_mask[:, :, f]
        if feature_mask.sum() > 0:
            feature_correct = masked_equalities[:, :, f][feature_mask].sum()
            feature_acc = feature_correct.float() / feature_mask.sum().float()
            results.append(
                {
                    "metric": f"Feature_{f}_Accuracy",
                    "value": f"{feature_acc.item():.6f}",
                }
            )

    return pd.DataFrame(results)


def main(epoch_num: int, batch_size: int = 32):
    """Main evaluation pipeline with batched processing."""
    print(f"Starting comprehensive model evaluation with batch size {batch_size}...")

    # Load model and dataset
    model, dataset = load_model_and_dataset(epoch_num)

    results = []
    all_targets = []
    all_generated = []

    # Create batches of word indices
    batches = create_batches(len(dataset), batch_size)

    print(f"Processing {len(dataset)} words in {len(batches)} batches...")

    # Process each batch
    for batch_indices in tqdm(batches, desc="Processing batches"):
        batch_results, batch_targets, batch_generated = evaluate_batch_of_words(
            model, batch_indices, dataset, batch_size
        )

        # Accumulate results
        results.extend(batch_results)
        all_targets.extend(batch_targets)
        all_generated.extend(batch_generated)

    print(f"Successfully evaluated {len(results)} words.")

    # Create DataFrames
    df_words = pd.DataFrame(results)
    df_summary = calculate_dataset_metrics(all_targets, all_generated)

    # Save to Excel with multiple sheets
    output_file = f"p2p_english_learner_evaluation_epoch_{epoch_num}_batched.xlsx"

    print(f"Saving results to {output_file}...")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_words.to_excel(writer, sheet_name="Per_Word_Results", index=False)
        df_summary.to_excel(writer, sheet_name="Dataset_Summary", index=False)

    # Also save as CSV for easier programmatic access
    df_words.to_csv(f"per_word_results_epoch_{epoch_num}_batched.csv", index=False)
    df_summary.to_csv(f"dataset_summary_epoch_{epoch_num}_batched.csv", index=False)

    print("Evaluation complete!")
    print(f"Results saved to {output_file}")
    print(f"Per-word results: {len(df_words)} rows, {len(df_words.columns)} columns")
    print(f"Summary statistics: {len(df_summary)} metrics")

    # Print quick overview - FIXED: Show proper precision
    print("\n=== QUICK OVERVIEW ===")
    if len(results) > 0:
        word_acc = df_words["word_exact_match"].mean()
        feat_acc = df_words["overall_feature_accuracy"].mean()
        phon_acc = df_words["overall_phoneme_accuracy"].mean()
        avg_edit = df_words["edit_distance"].mean()

        print(f"Word Exact Match Rate: {word_acc:.6f}")
        print(f"Average Feature Accuracy: {feat_acc:.6f}")
        print(f"Average Phoneme Accuracy: {phon_acc:.6f}")
        print(f"Average Edit Distance: {avg_edit:.6f}")


if __name__ == "__main__":
    # You can adjust batch_size based on your GPU memory
    # Start with 32, increase to 64 or 128 if you have enough memory
    BATCH_SIZE = 64

    for epoch in range(49, 809, 50):
        print(f"\n=== Evaluating epoch {epoch} with batch size {BATCH_SIZE} ===")
        main(epoch_num=epoch, batch_size=BATCH_SIZE)
