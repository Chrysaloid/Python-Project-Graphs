import matplotlib
import importlib

print("Interactive backends:", matplotlib.backends.backend_registry.list_builtin(matplotlib.backends.BackendFilter.INTERACTIVE))
print("Non-interactive backends:", matplotlib.backends.backend_registry.list_builtin(matplotlib.backends.BackendFilter.NON_INTERACTIVE))
print("All backends:", matplotlib.backends.backend_registry.list_builtin())

usable_backends = []
for backend in matplotlib.backends.backend_registry.list_builtin():
	try:
		matplotlib.use(backend, force=True)
		importlib.import_module(matplotlib.rcsetup._validate_backend(backend))
		usable_backends.append(backend)
	except Exception:
		pass

print("Usable backends:", usable_backends)

print("Current backend:", matplotlib.get_backend())
