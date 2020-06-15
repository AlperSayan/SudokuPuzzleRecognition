
import glob
import operator
import os
import cv2 as cv
import natsort
import numpy as np


# this program approximately takes 4-5 minutes to finish


def part1(image, size):

    def hough_transform(image, hough_image):
        lines = cv.HoughLines(hough_image, 1, np.pi / 180, 150)
        if lines is None:
            return image

        if filter:
            rho_threshold = 15
            theta_threshold = 0.1

            # how many lines are similar to a given one
            similar_lines = {i: [] for i in range(len(lines))}
            for i in range(len(lines)):
                for j in range(len(lines)):
                    if i == j:
                        continue

                    rho_i, theta_i = lines[i][0]
                    rho_j, theta_j = lines[j][0]
                    if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                        similar_lines[i].append(j)

            # ordering the INDECES of the lines by how many are similar to them
            indices = [i for i in range(len(lines))]
            indices.sort(key=lambda x: len(similar_lines[x]))

            # line flags is the base for the filtering
            line_flags = len(lines) * [True]
            for i in range(len(lines) - 1):
                # if we already disregarded the ith element in the ordered list then we don't care
                if not line_flags[indices[i]]:
                    continue

                for j in range(i + 1, len(lines)):  # we are only considering those elements that had less similar line
                    if not line_flags[indices[j]]:  # and only if we have not disregarded them already
                        continue

                    rho_i, theta_i = lines[indices[i]][0]
                    rho_j, theta_j = lines[indices[j]][0]
                    if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                        line_flags[
                            indices[j]] = False  # if it is similar and have not been disregarded yet then drop it now

        filtered_lines = []

        if filter:
            for i in range(len(lines)):  # filtering
                if line_flags[i]:
                    filtered_lines.append(lines[i])
        else:
            filtered_lines = lines

        for line in filtered_lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)

            cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

        return image

    def return_rects(in_img, rects, colour=(0, 0, 255)):
        img = in_img.copy()
        for rect in rects:
            img = cv.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), colour, thickness=3)
        return img

    def preprocessor(image):
        preprocess = cv.GaussianBlur(image, (9, 9), 0)
        preprocess = cv.adaptiveThreshold(preprocess, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        preprocess = cv.bitwise_not(preprocess, preprocess)
        kernel = np.ones((5, 5), np.uint8)
        preprocess = cv.dilate(preprocess, kernel)
        return preprocess

    def preprocessor2(image, image_Size):
        if image_Size > 100:
            image = cv.GaussianBlur(image, (9, 9), 0)
        else:
            image = cv.GaussianBlur(image, (3, 3), 0)
        image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        image = cv.bitwise_not(image, image)

        kernel = np.ones((3, 3), np.uint8)
        edges = cv.dilate(image, kernel, iterations=1)
        kernel = np.ones((6, 6), np.uint8)
        edges = cv.erode(edges, kernel, iterations=1)
        return edges

    def get_largest_corners(image):
        contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        polygon = contours[0]

        bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        bottom_left, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_right, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

        return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

    def crop_sudoku_puzzle(image, crop_rectangle):
        img = image
        crop_rect = crop_rectangle

        def distance_between(a, b):
            return np.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2))

        def crop_img():
            top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

            source_rect = np.array(np.array([top_left, bottom_left, bottom_right, top_right], dtype='float32'))

            sides = max([
                distance_between(bottom_right, top_right),
                distance_between(top_left, bottom_left),
                distance_between(bottom_right, bottom_left),
                distance_between(top_left, top_right)
            ])

            dest_square = np.array([[0, 0], [sides - 1, 0], [sides - 1, sides - 1], [0, sides - 1]], dtype='float32')

            modified = cv.getPerspectiveTransform(source_rect, dest_square)

            return cv.warpPerspective(img, modified, (int(sides), int(sides)))

        return crop_img()

    image_to_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    if size > 100:
        preprocess = preprocessor(image_to_gray)
        corners_of_sudoku = get_largest_corners(preprocess)

    else:
        preprocess = preprocessor(image_to_gray)
        corners_of_sudoku = get_largest_corners(preprocess)

    cropped_image = crop_sudoku_puzzle(image, corners_of_sudoku)

    cropped_image_to_gray = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
    preprocess2 = preprocessor2(cropped_image_to_gray, size)

    if size < 100:
        canny_image = cv.Canny(preprocess2, 1, 100, apertureSize=3)

    elif 100 < size < 500:
        canny_image = cv.Canny(preprocess2, 90, 150, apertureSize=3)

    else:
        canny_image = cv.Canny(preprocess2, 30, 60, apertureSize=3)

    hough_image = hough_transform(cropped_image, canny_image)
    grids = infer_grid(hough_image)
    rects = return_rects(hough_image, grids)

    return rects


def get_int(byte):
    return int.from_bytes(byte, "big")


def infer_grid(img):  # infer 81 cells from image
    squares = []
    side = img.shape[:1]

    side = side[0] / 9

    for i in range(9):  # get each box and append it to squares -- 9 rows, 9 cols
        for j in range(9):
            p1 = (i * side, j * side)  # top left corner of box
            p2 = ((i + 1) * side, (j + 1) * side)  # bottom right corner of box
            squares.append((p1, p2))
    return squares


# def save_minst(data_dict):
#     minst_output_path = "MNISTData/"
#     sets = ['train', 'test']
#     for set in sets:
#         images = data_dict[set + '_images']
#         labels = data_dict[set + '_labels']
#         no_of_samples = images.shape[0]
#         for indx in range(no_of_samples):
#             print(set, indx)
#             image = images[indx]
#             label = labels[indx]
#             if not os.path.exists(minst_output_path + set + '/' + str(label) + '/'):
#                 os.makedirs(minst_output_path + set + '/' + str(label) + '/')
#
#             file_number = len(os.listdir(minst_output_path + set + '/' + str(label) + '/'))
#             cv.imwrite(minst_output_path + set + '/' + str(label) + '/%05d.png' % (file_number), image)


def load_minst(minst_dataset):
    data_dict = {}

    for file_name in minst_dataset:
        if file_name.endswith('ubyte'):
            with open(file_name, 'rb') as f:
                data = f.read()
                type = get_int(data[:4])
                length = get_int(data[4:8])
                if (type == 2051):
                    category = 'images'
                    num_rows = get_int(data[8:12])
                    num_cols = get_int(data[12:16])
                    parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
                    parsed = parsed.reshape(length, num_rows, num_cols)
                elif (type == 2049):
                    category = 'labels'
                    parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
                    parsed = parsed.reshape(length)
                if (length == 10000):
                    set = 'test'
                elif (length == 60000):
                    set = 'train'
                data_dict[set + '_' + category] = parsed

    return data_dict

# to remove hough lines around corners
def preprocess_image(img):
    img = np.array(img)
    rows = np.shape(img)[0]

    # First remove the outermost white pixels.
    # This can be achieved by flood filling with some of the outer points

    for i in range(rows):
        # Floodfilling the outermost layer
        cv.floodFill(img, None, (0, i), 0)
        cv.floodFill(img, None, (i, 0), 0)
        cv.floodFill(img, None, (rows - 1, i), 0)
        cv.floodFill(img, None, (i, rows - 1), 0)
        # Floodfilling the second outermost layer
        cv.floodFill(img, None, (1, i), 1)
        cv.floodFill(img, None, (i, 1), 1)
        cv.floodFill(img, None, (rows - 2, i), 1)
        cv.floodFill(img, None, (i, rows - 2), 1)

    # Finding the bounding box of the number in the cell
    rowtop = None
    rowbottom = None
    colleft = None
    colright = None
    thresholdBottom = 100
    thresholdTop = 100
    thresholdLeft = 100
    thresholdRight = 100
    center = rows // 2
    for i in range(center, rows):
        if rowbottom is None:
            temp = img[i]
            if sum(temp) < thresholdBottom or i == rows - 1:
                rowbottom = i
        if rowtop is None:
            temp = img[rows - i - 1]
            if sum(temp) < thresholdTop or i == rows - 1:
                rowtop = rows - i - 1
        if colright is None:
            temp = img[:, i]
            if sum(temp) < thresholdRight or i == rows - 1:
                colright = i
        if colleft is None:
            temp = img[:, rows - i - 1]
            if sum(temp) < thresholdLeft or i == rows - 1:
                colleft = rows - i - 1

    # Centering the bounding box's contents
    newimg = np.zeros(np.shape(img))
    startatX = (rows + colleft - colright) // 2
    startatY = (rows - rowbottom + rowtop) // 2
    for y in range(startatY, (rows + rowbottom - rowtop) // 2):
        for x in range(startatX, (rows - colleft + colright) // 2):
            newimg[y, x] = img[rowtop + y - startatY, colleft + x - startatX]

    return newimg


def SudokuDigitDetector(cells, trained):
    cells = get_cells(cells)

    for i in range(len(cells)):
        for j in range(len(cells[0])):

            cells[i][j] = preprocess_image(cells[i][j])

    cells = resize_cells(cells)

    cells = np.array(cells)
    cells = cells.reshape([81, 28, 28])
    cells_pca = pca_(cells, 3)

    ret, results, neighbours, dist = trained.findNearest(cells_pca, k=10)

    for i in range(len(results)):
        white_cells = np.count_nonzero(cells[i])
        if white_cells < 50:
            results[i] = 0

    results = results.reshape(9, 9)

    return results


def sudokuAcc(gt, out):
    return (gt == out).sum() / gt.size * 100


def pca_(images, numpc):
    np.seterr(divide='ignore', invalid='ignore')
    pcas = []
    for i in range(len(images)):
        img = images[i]
        img = img - np.mean(img, 0)
        img = img / np.std(img, 0)
        img = np.nan_to_num(img)
        C = np.dot(img.T, img) / img.shape[0]
        E, V = np.linalg.eigh(C)
        key = np.argsort(E)[::-1][:numpc]
        E, V = E[key], V[:, key]
        U = np.dot(images[i], V)  # new coordinates

        pcas.append(U)

    pcas = np.array(pcas)
    pcas_1d = np.reshape(pcas, (pcas.shape[0], pcas.shape[1] * pcas.shape[2]))
    pcas_1d = pcas_1d.astype('float32')

    return pcas_1d


def compute_confusion_matrix(true, pred):
    K = len(np.unique(true))  # Number of classes
    result = np.zeros((K, K))

    for i in range(len(true)):
        result[true[i]][pred[i]] += 1

    return result


# def draw_conf_matrix(conf_matrix):
#    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
#
#    fig, ax = plt.subplots()
#    im = ax.imshow(conf_matrix)
#
#    ax.set_xticks(np.arange(len(classes)))
#    ax.set_yticks(np.arange(len(classes)))
#    # ... and label them with the respective list entries
#    ax.set_xticklabels(classes)
#    ax.set_yticklabels(classes)
#
#    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#             rotation_mode="anchor")
#
#    for i in range(len(classes)):
#        for j in range(len(classes)):
#            text = ax.text(j, i, conf_matrix[i, j],
#                           ha="center", va="center", color="w")
#
#    ax.set_title("Confusion matrix of the Sudoku dataset")
#    fig.tight_layout()
#    plt.show()


def train_MINST(minst_dict):
    def report_accuracy(result, labels):
        sum = 0
        for i in range(len(result)):
            if labels[i] == result[i][0]:
                sum += 1
        return sum / len(result)

    train_images = np.copy(minst_dict['train_images'])
    train_labels = np.copy(minst_dict['train_labels']).astype('float32')
    test_images = np.copy(minst_dict['test_images'])
    test_labels = minst_dict['test_labels']

    knn = cv.ml.KNearest_create()

    x_train_pca = pca_(train_images, 3)
    y_train_pca = pca_(test_images, 3)

    knn.train(x_train_pca, cv.ml.ROW_SAMPLE, train_labels)

    ret, results, neighbours, dist = knn.findNearest(y_train_pca, k=10)

    # results_conf = results.flatten()
    # conf_matrix = compute_confusion_matrix(test_labels.astype('uint8'), results_conf.astype('uint8'))
    # draw_conf_matrix(conf_matrix)

    print("Accuracy of MNIST dataset test of " + str(len(test_labels)) + " test items is: %" + str(
        int(report_accuracy(results, test_labels) * 100)))
    print("Mean distance " + str(int(np.mean(dist))))

    return knn


# make sudoku cells (28*28) pixels
def resize_cells(cells):
    for i in range(len(cells)):
        for j in range(len(cells[0])):
            image = cells[i][j]
            cells[i][j] = cv.resize(np.float32(image), (28, 28))
    return cells


# get 81 cells from a sudoku grid
def get_cells(grid):
    grid = np.copy(cv.cvtColor(grid, cv.COLOR_BGR2GRAY))
    edge = np.shape(grid)[0]
    celledge = edge // 9

    # Adaptive thresholding the cropped grid and inverting it

    grid = cv.bitwise_not(cv.adaptiveThreshold(grid, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 1))

    # Creating a vector of size 81 of all the cell images
    tempgrid = []
    for i in range(celledge, edge + 1, celledge):
        for j in range(celledge, edge + 1, celledge):
            rows = grid[i - celledge:i]
            tempgrid.append([rows[k][j - celledge:j] for k in range(len(rows))])

        # Creating the 9X9 grid of images
    finalgrid = []
    for i in range(0, len(tempgrid) - 8, 9):
        finalgrid.append(tempgrid[i:i + 9])

    # Converting all the cell images to np.array
    for i in range(9):
        for j in range(9):
            finalgrid[i][j] = np.array(finalgrid[i][j])

        return finalgrid


if __name__ == "__main__":

    # MNIST experiments:

    mints_dataset_file_names = ["train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte",
                                "t10k-labels.idx1-ubyte"]
    MINST_dict = load_minst(mints_dataset_file_names)
    trained = train_MINST(MINST_dict)

    # Sudoku Experiments:

    image_dirs = 'images/*.jpg'
    data_dirs = 'images/*.dat'
    IMAGE_DIRS = natsort.natsorted(glob.glob(image_dirs))
    DATA_DIRS = natsort.natsorted(glob.glob(data_dirs))
    total_acc = 0

    # Loop over all images and ground truth
    #conf_matrix = np.zeros((10, 10))

    for i, (img_dir, data_dir) in enumerate(zip(IMAGE_DIRS, DATA_DIRS)):

        image_name = os.path.basename(img_dir)
        image_size = os.path.getsize(img_dir) / 1000
        gt = np.genfromtxt(data_dir, skip_header=2, dtype=int, delimiter=' ')
        img = cv.imread(img_dir)
        sudoku = part1(img, image_size)  # outputs sudoku grid with houghlines

        output = SudokuDigitDetector(sudoku, trained)

        output_1d = output.flatten()
        gt_1d = gt.flatten()

        #comp = compute_confusion_matrix(gt_1d.astype('uint8'), output_1d.astype('uint8'))
        #if comp.shape == conf_matrix.shape:
            #conf_matrix = conf_matrix + comp

        total_acc = total_acc + sudokuAcc(gt, output)

    #draw_conf_matrix(conf_matrix)

    print("Sudoku dataset accuracy: {}".format(total_acc / (i + 1)))
