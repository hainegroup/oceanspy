# Import modules
import subprocess


# Download data
def pytest_configure():
    print("\nHello! I'm downloading test data.")

    # Directory
    Datadir = './oceanspy/tests/'

    # Download xmitgcm test
    commands = ['cd {}'.format(Datadir),
                'rm -fr Data',
                'wget -v -O Data.tar.gz -L '
                'https://jh.box.com/'
                'shared/static/lezaefccn11zmbrvtvollmzgkbw9r8ie.gz',
                'tar xvzf Data.tar.gz',
                'rm -f Data.tar.gz']
    subprocess.call('&&'.join(commands), shell=True)
