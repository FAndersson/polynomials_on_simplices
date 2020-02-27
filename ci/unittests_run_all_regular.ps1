# Run library unit tests, on a Windows machine.
# Utilizes Docker to run the Linux script run_unittests.sh in a standardized environment.
#
# Dependencies:
# * Docker.

# Start Docker container
$libraryPath = (get-item $PSScriptRoot).parent.FullName
docker run --rm --tty --interactive --detach --volume ${libraryPath}:/repository --name run-unittests-container fredrikandersson/debian-stable-python-image:stable-2019-12-24 /bin/bash

# Clone the repository to a new folder in the Docker container
docker exec run-unittests-container /bin/bash -c "git clone /repository /tmp/polynomials_on_simplices"

# Run unit tests in the Docker container
docker exec run-unittests-container /bin/bash -c "cd /tmp/polynomials_on_simplices/ci; chmod 755 unittests_run_all_regular.sh; ./unittests_run_all_regular.sh"

# Kill Docker container
docker kill run-unittests-container
