import importlib


def load_file(filename):
  def make_module_path(s):
    ''' Convert apossible filepath to a module-path. Does nothing it s is already a module-path '''
    return s.replace('.py', '').replace('/', '.').replace('..', '.').lstrip('.')
  filename = importlib.import_module(make_module_path(filename))
  meta = filename.meta if hasattr(filename, 'meta') else None
  cb = filename.matplot_network_writer_hook if hasattr(filename, 'matplot_network_writer_hook') else None
  return (filename.make_deviceset(), meta, cb)
