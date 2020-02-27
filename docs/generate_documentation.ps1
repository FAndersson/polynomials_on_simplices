# Generate library documentation from docstrings using sphinx, on a Windows machine.
# Utilizes Docker to run the Linux script generate_documentation.sh in a standardized environment.
#
# Dependencies:
# * Docker.

# Start Docker container
$libraryPath = (get-item $PSScriptRoot).parent.FullName
docker run --rm --tty --interactive --detach `
--volume ${libraryPath}:/repository `
--name generate-documentation-container `
fredrikandersson/debian-stable-python-image /bin/bash

# Clone the repository to a new folder in the Docker container
docker exec generate-documentation-container `
/bin/bash -c "git clone /repository /tmp/polynomials_on_simplices"

# Generate documentation in the Docker container
docker exec generate-documentation-container `
/bin/bash -c `
"cd /tmp/polynomials_on_simplices/docs; chmod 755 generate_documentation.sh; ./generate_documentation.sh"

# Pack generated documentation
docker exec generate-documentation-container `
/bin/bash -c `
"cd /tmp/polynomials_on_simplices/docs; chmod 755 zip_generated_documentation.sh; ./zip_generated_documentation.sh"

# Copy zipped generated documentation from the docker container to the script directory on the host machine
$version = git describe --always
$container_path = "/tmp/polynomials_on_simplices/docs/generated_documentation/polynomials_on_simplices_documentation_$version.zip"
If(!(test-path generated_documentation))
{
      New-Item -ItemType Directory -Force -Path generated_documentation
}
docker cp generate-documentation-container:${container_path} generated_documentation\

# Kill Docker container
docker kill generate-documentation-container
