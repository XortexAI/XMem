import unittest
from unittest.mock import patch
from src.scanner.git_ops import clone_repo, pull_latest

class TestGitOps(unittest.TestCase):
    def test_clone_repo_rejects_options(self):
        with self.assertRaisesRegex(ValueError, "Invalid repo_url: cannot start with '-'"):
            clone_repo("--upload-pack=malicious", "/tmp/local", "main")

        with self.assertRaisesRegex(ValueError, "Invalid branch: cannot start with '-'"):
            clone_repo("https://github.com/repo.git", "/tmp/local", "--branch")

    def test_pull_latest_rejects_options(self):
        with self.assertRaisesRegex(ValueError, "Invalid branch: cannot start with '-'"):
            pull_latest("/tmp/local", "--branch")

    @patch("src.scanner.git_ops._run_git")
    @patch("src.scanner.git_ops.get_head_sha")
    @patch("src.scanner.git_ops.Path")
    def test_clone_repo_happy_path(self, mock_path, mock_get_head, mock_run_git):
        mock_get_head.return_value = "12345678"
        sha = clone_repo("https://github.com/repo.git", "/tmp/local", "main")
        self.assertEqual(sha, "12345678")

        mock_run_git.assert_called_with(
            ["clone", "--branch", "main", "--single-branch", "--", "https://github.com/repo.git", "/tmp/local"],
            cwd=mock_path.return_value.parent.__str__.return_value,
            timeout=1800
        )

    @patch("src.scanner.git_ops._run_git")
    @patch("src.scanner.git_ops.get_head_sha")
    def test_pull_latest_happy_path(self, mock_get_head, mock_run_git):
        mock_get_head.return_value = "87654321"
        sha = pull_latest("/tmp/local", "main")
        self.assertEqual(sha, "87654321")

        # Verify checkout call
        mock_run_git.assert_any_call(["checkout", "main"], cwd="/tmp/local")
        # Verify pull call
        mock_run_git.assert_any_call(["pull", "origin", "main"], cwd="/tmp/local", timeout=600)

if __name__ == '__main__':
    unittest.main()
