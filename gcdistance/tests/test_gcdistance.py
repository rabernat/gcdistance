import numpy as np
import pytest

from gcdistance.gcdistance import great_circle_distance

def test_great_circle_distance():
    # some known results
    # distance between two same points should be zero
    assert great_circle_distance((20., 30.), (20., 30.)) == 0

    # check distance between new york and london
    new_york = 40.7128, -74.0060
    london = 51.5074, 0.1278
    dist_nyc_london = great_circle_distance(new_york, london)
    # very strict
    # assert dist_nyc_london == 5.587e6
    np.testing.assert_allclose(dist_nyc_london, 5.587e6, rtol=1e-5)

    # now check that we can't pass the wrong number of arguments
    with pytest.raises(TypeError):
        great_circle_distance(1, 2, 3, 4)
