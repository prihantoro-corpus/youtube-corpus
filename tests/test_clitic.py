import sys
import os

# Add parent directory to path to import indonesian_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import indonesian_utils

def test_tokenize_word():
    # Test cases from rules
    
    # Rule 0: Short words
    assert indonesian_utils.tokenize_word("aku") == ["aku"]
    assert indonesian_utils.tokenize_word("dia") == ["dia"]
    
    # Rule 1: Proklitik ku-
    assert indonesian_utils.tokenize_word("kubawa") == ["ku", "bawa"]
    assert indonesian_utils.tokenize_word("Kuliah") == ["Kuliah"] # Exception BLOCK_KU_PREFIX
    assert indonesian_utils.tokenize_word("kucinta") == ["ku", "cinta"]
    
    # Case preservation Rule 1
    assert indonesian_utils.tokenize_word("Kubawa") == ["Ku", "bawa"] 
    # Wait, code: prefix_tokens.append(W[:2]). W[:2] is "Ku". W becomes "bawa". correct.
    
    # Rule 2: Enklitik -nya
    assert indonesian_utils.tokenize_word("bukunya") == ["buku", "nya"]
    assert indonesian_utils.tokenize_word("semuanya") == ["semua", "nya"] # PRONOMINAL_SE
    assert indonesian_utils.tokenize_word("harusnya") == ["harus", "nya"]
    
    # Case preservation Rule 2
    assert indonesian_utils.tokenize_word("Bukunya") == ["Buku", "nya"]
    
    # Rule 3: Enklitik -ku / -mu
    assert indonesian_utils.tokenize_word("badanku") == ["badan", "ku"]
    assert indonesian_utils.tokenize_word("Badanmu") == ["Badan", "mu"]
    assert indonesian_utils.tokenize_word("ilmu") == ["ilmu"] # Exception
    assert indonesian_utils.tokenize_word("tamu") == ["tamu"] # Exception
    
    # Combined / Recursive? The current logic is sequential.
    # W is updated.
    # Example: "kuharapnya" -> Rule 1 "ku", "harapnya" -> Rule 2 "harap", "nya".
    # Result: "ku", "harap", "nya".
    # Let's see code.
    # Rule 1 modifies W. Rule 2 checks W.
    assert indonesian_utils.tokenize_word("kuharapnya") == ["ku", "harap", "nya"]

def test_tokenize_text():
    text = "Aku membawa bukunya ke rumahmu."
    expected = ["Aku", "membawa", "buku", "nya", "ke", "rumah", "mu", "."]
    assert indonesian_utils.tokenize_text(text) == expected
    
    text2 = "Badanmu sehat."
    assert indonesian_utils.tokenize_text(text2) == ["Badan", "mu", "sehat", "."]
    
    print("All tests passed!")

if __name__ == "__main__":
    test_tokenize_word()
    test_tokenize_text()
