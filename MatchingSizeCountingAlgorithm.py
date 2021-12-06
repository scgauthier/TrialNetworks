# Total number of edges = N(N-1)/2

def count_size_2_matchings(NumUsers: int) -> int:

    Total_matchings = 0
    Total_edges = (NumUsers * (NumUsers - 1)) / 2

    # x is Ca
    for x in range(1, NumUsers - 1):
        for y in range(x, NumUsers - 1):

            backlog = 0
            for b in range(1, x):
                backlog += NumUsers - b - 1  # -1 prevents double counting
            print(backlog)
            Total_matchings += (Total_edges
                                - (NumUsers - x)
                                - backlog
                                - y + 1)  # +1 prevents double counting
            print(Total_matchings)

    return Total_matchings


# Note with this algorithm, can prove that the number of matchings of size 2
# when the number of users is N is given by the sum from 1 to Am, where Am is
# given by the series A0 = 1, Am = A(m-1) + (m + 1) and m=(N-3). Hence the
# number of matchings  of size 2 is Am(Am + 1)/2 or (N-3)(N-2)/2
