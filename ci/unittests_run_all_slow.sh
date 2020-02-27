# The library root directory need to be part of the python path
export PYTHONPATH=$PYTHONPATH:..
# Run only slow unit tests using pytest
pytest -c pytest.ini --doctest-modules -m "slow" --durations=10 --ignore=../polynomials_on_simplices/generic_tools/test/test_package ../