import logging
import os
from urllib.request import urlopen


def download_data(url, outfile='', binary=False):
    """
    Downloads data_util from a URL and returns raw data_util.

    Parameters
    ----------
    url : str
        URL to get the data_util from
    outfile : str, optional
        Where to save the data_util.
    binary : bool, optional
        If true, writes data_util in binary.
    """
    # Try statement to catch downloads.
    try:
        # Download url using urlopen
        with urlopen(url) as f:
            data = f.read()
        logging.info('Data download success!')
        success = True
    except:
        logging.info('Data download failed!')
        success = False

    if binary:
        # If data_util is binary, use 'wb' if outputting to file
        writeflag = 'wb'
    else:
        # If data_util is string, convert to string and use 'w' if outputting to file
        writeflag = 'w'
        data = data.decode('utf-8')

    if outfile:
        logging.info('Saving downloaded data_util to file: {}'.format(outfile))

        with open(outfile, writeflag) as f:
            f.write(data)

    return data, success


# Check if a string can be converted to an int, without throwing an error.
def is_int(str):
    try:
        int(str)
        return True
    except:
        return False

# Cleanup. Use try-except to avoid race condition.
def cleanup_file(file, cleanup=True):
    if cleanup:
        try:
            os.remove(file)
        except OSError:
            pass
