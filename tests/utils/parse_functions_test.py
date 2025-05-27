import pytest

from utils.parse_functions import (extract_from_basemodel,
                                   parse_output_to_basemodel)

@pytest.fixture
def sample_basemodel():
    output = {"car_id": 5,
              "car_make": [{"car_brand": "BMC", "colors": [{"rgb": "red"}, 
                                                           {"rgb": "green"}]}, 
                           {"car_brand": "BMC", "colors": [{"rgb": "blue"},
                                                           {"rgb": "black"}]}]}
    type_repr = parse_output_to_basemodel(output, "bob")
    return type_repr(**output)

@pytest.mark.parametrize(
        "output_builder, field_path, expected_value, expected_sig",
        [
            pytest.param("sample_basemodel", "car_make.car_brand", ["BMC", "BMC"], "car_make.car_brand@d",
                         id="flat_car_brand"),
            pytest.param("sample_basemodel", "car_make.colors.rgb", [["red", "green"], ["blue", "black"]], "car_make.colors@d.rgb@d",
                         id="nested_colors_rgb")
        ]
)
def test_extract_from_basemodel(output_builder, field_path, expected_value, expected_sig, request):
    output = request.getfixturevalue(output_builder)
    value, sig = extract_from_basemodel(output, field_path)

    assert value == expected_value
    assert sig == expected_sig

