"""
Update version in conf.py based on current Git commit/tag.
"""

import subprocess


def get_git_description():
    """
    Get description of current Git commit (tag or commit hash in short format).

    This is the output from 'git describe --always'.

    :return: Description of current Git commit. Returns None if this is not a Git repository.
    :rtype: Optional[str].
    """
    # Try running 'git describe --always'
    ret = subprocess.call(["git", "describe", "--always"], stdout=subprocess.DEVNULL)
    if ret == 0:
        # Running 'git describe --always' was successful. Return output as string
        return subprocess.check_output(["git", "describe", "--always"]).decode("utf-8")[0:-1]
    # 'git describe' failed. Probably not a Git repository
    return None


def update_version_in_conf():
    """
    Update the version entry in conf.py using the current version from Git (from :func:`get_git_description`).
    """
    version = get_git_description()
    if version is None:
        return

    with open('conf.py') as file:
        file_content = file.read()
    file_content = file_content.replace("version = ''", "version = '" + version + "'")
    with open('conf.py', 'w') as file:
        file.write(file_content)


if __name__ == "__main__":
    update_version_in_conf()
