import weave

from bufogen.rendering.svg import validate_svg


class SvgValidity(weave.Scorer):
    @weave.op()
    def score(self, output: dict) -> dict:
        valid, error = validate_svg(output["svg"])
        return {
            "valid": bool(valid and output.get("valid", False)),
            "error": error or output.get("error"),
        }
