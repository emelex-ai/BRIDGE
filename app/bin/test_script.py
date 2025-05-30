import pickle
import torch

with open("results/pretraining_1_epoch_0.pkl", "rb") as f:
    data = pickle.load(f)

# First clone the tensors to avoid modifying the originals
phon_preds = data.pretraining.phon_predictions.clone()
phon_targets = data.pretraining.phon_targets.clone()

# ===== FEATURE LEVEL ACCURACY =====
# Create mask for valid features (not padding tokens which are marked as 2)
phon_features_mask = phon_targets != 2

# Find which predictions match targets, but only count valid features
masked_equalities = torch.eq(phon_preds, phon_targets) & phon_features_mask

# Calculate overall feature accuracy across entire dataset
feature_accuracy = masked_equalities.sum() / phon_features_mask.sum()
print("\nOverall Feature-Level Accuracy:", feature_accuracy.item())

# ===== PHONEME LEVEL ACCURACY =====
# A phoneme is correct only if ALL its features are correct
# We use .all(dim=2) to check across feature dimension
phoneme_correct = masked_equalities.all(dim=2)

# Identify valid phonemes (those where not all features are 2/padding)
valid_phonemes = ~(phon_targets == 2).all(dim=2)

# Calculate overall phoneme accuracy
phoneme_accuracy = phoneme_correct[valid_phonemes].sum() / valid_phonemes.sum()
print("Overall Phoneme-Level Accuracy:", phoneme_accuracy.item())

# ===== WORD LEVEL ACCURACY =====
# A word is correct only if ALL its valid phonemes are correct
word_correct = torch.all(phoneme_correct | ~valid_phonemes, dim=1)
word_accuracy = word_correct.sum() / float(word_correct.size(0))
print("Overall Word-Level Accuracy:", word_accuracy.item())

# ===== PER-WORD DETAILED ANALYSIS =====
print("\nDetailed per-word analysis:")
for i in range(min(5, len(word_correct))):  # Show first 5 words as example
    # Count valid features for this word
    n_valid_features = phon_features_mask[i].sum().item()
    n_correct_features = masked_equalities[i].sum().item()

    # Count valid phonemes for this word
    n_valid_phonemes = valid_phonemes[i].sum().item()
    n_correct_phonemes = (phoneme_correct[i] & valid_phonemes[i]).sum().item()

    print(f"\nWord {i}:")
    print(f"Features: {n_correct_features}/{n_valid_features} correct")
    print(f"Phonemes: {n_correct_phonemes}/{n_valid_phonemes} correct")
    print(f"Word-level: {'Correct' if word_correct[i] else 'Incorrect'}")
