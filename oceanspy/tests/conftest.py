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
                'https://livejohnshopkins-my.sharepoint.com/'
                ':u:/g/personal/malmans2_jh_edu/'
                'ERLGPTpYMyRWmGW0YqVyFaQBiHxDuDzSMQPC4_UfH5e5Qg'
                '?download=1',
                'tar xvzf Data.tar.gz',
                'rm -f Data.tar.gz']
    subprocess.call('&&'.join(commands), shell=True)
