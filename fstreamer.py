def stream(filename, index_file, last_index_position):
    """
    This function retrieves a line from the dataset by referencing a position stored in an index file. 
    It maintains the last read position to enable continuous streaming.

    Parameters
    ----------
    filename : str
        The path to the dataset file to be read.
    index_file : str
        The path to the index file containing byte positions of lines in the dataset.
    last_index_position : int
        The last read position in the index file.

    Returns
    -------
    tuple
        A tuple containing:
        - str: The extracted line from the dataset file (or "end" if EOF is reached).
        - int or str: The updated last index position (or "end" if EOF is reached).
    """

    f_indexed_lines = open(index_file)
    f_dataset = open(filename, "rb")

    f_indexed_lines.seek( last_index_position )
    position_raw = f_indexed_lines.readline()
    
    last_index_position += len(position_raw)
    
    position = position_raw.split('\t')
    if position == ['']:
        return "end", "end"
    
    pos_beg = int(position[0])
    f_dataset.seek( pos_beg )
    line = f_dataset.readline()
    line = line.decode("utf-8", errors="ignore")
    line = line.strip()

    return line, last_index_position
