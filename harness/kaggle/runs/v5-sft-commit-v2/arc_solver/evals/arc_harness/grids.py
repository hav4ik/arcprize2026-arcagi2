"""ARC grid validity: the single definition used for answers, submissions and scoring."""


def valid_grid(grid):
    """A rectangular list of 1-30 rows of 1-30 Python ints in 0-9."""
    return (isinstance(grid, list) and 1 <= len(grid) <= 30
            and all(isinstance(row, list) and 1 <= len(row) <= 30 for row in grid)
            and len({len(row) for row in grid}) == 1
            and all(type(value) is int and 0 <= value <= 9 for row in grid for value in row))
