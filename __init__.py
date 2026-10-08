__all__ = ["PillPredictor", "predict"]


def __getattr__(name):
	if name in __all__:
		from .inference import PillPredictor, predict

		return {"PillPredictor": PillPredictor, "predict": predict}[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")