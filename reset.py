import os
import shutil

# Remove saved model
if os.path.exists('model'):
    shutil.rmtree('model')
    print('Model deleted')

# Remove plot
if os.path.exists('progress.png'):
    os.remove('progress.png')
    print('Plot deleted')

print('Ready for fresh training!')