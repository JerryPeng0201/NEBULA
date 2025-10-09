"""
Version information for NEBULA.

This file contains all version-related metadata for the NEBULA project.
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path

# ============================================================================
# Version Information
# ============================================================================
__version__ = "0.1.0"
__version_info__ = (0, 1, 0)

# ============================================================================
# Project Metadata
# ============================================================================
__title__ = "NEBULA-Alpha"
__description__ = "Advanced Robotic Manipulation Simulation Platform"
__author__ = "NEBULA Development Team (Jerry Peng, Yanyan Zhang, Yicheng Duan) from VU Lab"
__author_email__ = "jxp1146@case.edu"
__license__ = "GPL-3.0"
__copyright__ = f"Copyright 2025-{datetime.now().year}, NEBULA Development Team"
__url__ = "https://github.com/JerryPeng0201/NEBULA-Alpha#"
__documentation__ = None

# ============================================================================
# Git Information
# ============================================================================
def get_git_info():
    """
    Get current git commit hash and branch name.
    
    Returns:
        tuple: (commit_hash, branch_name) or ("unknown", "unknown") if not a git repo
    """
    try:
        git_dir = Path(__file__).parent.parent.parent / ".git"
        if not git_dir.exists():
            return "unknown", "unknown"
        
        # Get commit hash
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent,
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit = "unknown"
        
        # Get branch name
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent,
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            branch = "unknown"
        
        return commit, branch
    except Exception:
        return "unknown", "unknown"


# Populate git information
__git_commit__, __git_branch__ = get_git_info()

# Try to get git tag if available
try:
    __git_tag__ = subprocess.check_output(
        ["git", "describe", "--tags", "--exact-match"],
        stderr=subprocess.DEVNULL,
        cwd=Path(__file__).parent,
    ).decode().strip()
except (subprocess.CalledProcessError, FileNotFoundError):
    __git_tag__ = None

# ============================================================================
# Build Information
# ============================================================================
__build_date__ = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ============================================================================
# Feature Flags
# ============================================================================
__features__ = {
    "gpu_simulation": True,
    "sapien_render": True,
    "ai2thor_scenes": True,
    "real_robot_interface": True,
    "trajectory_replay": True,
    "vectorized_envs": True,
    "sb3_integration": True,
}

# ============================================================================
# Dependency Version Requirements
# ============================================================================
__requires__ = {
    "python": ">=3.8",
    "sapien": ">=3.0.0",
    "torch": ">=2.0.0",
    "gymnasium": ">=0.29.0",
    "numpy": ">=1.20.0",
}

# ============================================================================
# Version String Utilities
# ============================================================================
def get_version_string(include_git=True):
    """
    Get a formatted version string.
    
    Args:
        include_git: Whether to include git information
    
    Returns:
        str: Formatted version string
    """
    version_str = __version__
    
    if include_git and __git_commit__ != "unknown":
        version_str += f"+{__git_commit__}"
        if __git_branch__ != "unknown" and __git_branch__ not in ["main", "master"]:
            version_str += f".{__git_branch__}"
    
    return version_str


def get_full_version_info():
    """
    Get complete version information as a formatted string.
    
    Returns:
        str: Multi-line string with all version information
    """
    info_lines = [
        f"NEBULA Version Information",
        f"=" * 60,
        f"Version:        {__version__}",
        f"Git Commit:     {__git_commit__}",
        f"Git Branch:     {__git_branch__}",
        f"Git Tag:        {__git_tag__ or 'N/A'}",
        f"Build Date:     {__build_date__}",
        f"",
        f"Project Info:",
        f"  Title:        {__title__}",
        f"  Description:  {__description__}",
        f"  URL:          {__url__}",
        f"  License:      {__license__}",
        f"",
        f"Features:",
    ]
    
    for feature, enabled in __features__.items():
        status = "✓" if enabled else "✗"
        info_lines.append(f"  {status} {feature}")
    
    info_lines.append("=" * 60)
    
    return "\n".join(info_lines)


# ============================================================================
# Exports
# ============================================================================
__all__ = [
    "__version__",
    "__version_info__",
    "__title__",
    "__description__",
    "__author__",
    "__author_email__",
    "__license__",
    "__copyright__",
    "__url__",
    "__documentation__",
    "__git_commit__",
    "__git_branch__",
    "__git_tag__",
    "__build_date__",
    "__features__",
    "__requires__",
    "get_version_string",
    "get_full_version_info",
]

