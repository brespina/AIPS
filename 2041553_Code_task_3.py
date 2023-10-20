"""
Brandon Espina
09/29/2023
COSC 4368
Dr. Lin
"""


def is_valid(B, F, G, H, I):
    # Check the constraints
    if G ** 2 <= ((H - F) ** 2 + 4 - B - F) ** 2 + 694:
        return False

    if I > (H - F) ** 2 + 4 - B - F + 21:
        return False

    return True


def write_solution(B, F, G, H, I):
    # C can be any value between 1-125 since the only thing that depends on it is A and A can be set post search
    C = 1
    A = C + (H - F) ** 2 + 4
    D = 2 * ((H - F) ** 2) + 8 - 2 * B - 2 * F + 21
    E = (H - F) ** 2 + 4 - B - F
    result = (A, B, C, D, E, F, G, H, I)

    return result


def improved_BF_search():
    nva = 0
    for B in range(1, 125):
        nva += 1
        for F in range(1, 125):
            nva += 1
            for H in range(1, 125):
                nva += 1
                for I in range(1, 125):
                    nva += 1
                    for G in range(27, 125):
                        nva += 1
                        if is_valid(B, F, G, H, I):
                            sol = write_solution(B, F, G, H, I)
                            return sol, nva
    return "no solution"


solution = improved_BF_search()
print(
    f"result: A: {solution[0][0]} B: {solution[0][1]} C: {solution[0][2]} D: {solution[0][3]} E: {solution[0][4]} F: {solution[0][5]} G: {solution[0][6]} H: {solution[0][7]} I: {solution[0][8]}")
print(f"nva: {solution[1]}")
