# The library root directory need to be part of the python path
export PYTHONPATH=$PYTHONPATH:..
# Run unit tests using pytest, excluding slow tests
pytest -c pytest.ini --doctest-modules -m "not slow" --durations=10 --ignore=../polynomials_on_simplices/generic_tools/test/test_package ../