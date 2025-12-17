# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/nod-ai/amd-shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                 |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------- | -------: | -------: | ------: | --------: |
| amdsharktuner/amdsharktuner/\_\_init\_\_.py          |        7 |        0 |    100% |           |
| amdsharktuner/amdsharktuner/candidate\_gen.py        |      134 |       46 |     66% |36, 50, 55, 61, 72, 89-90, 100, 103, 116, 122, 137, 149, 162, 168, 175, 182, 190-191, 195, 201, 229-231, 251-256, 272-284, 294-299, 305-321 |
| amdsharktuner/amdsharktuner/candidate\_ordering.py   |       95 |       11 |     88% |89-92, 109-110, 179-187 |
| amdsharktuner/amdsharktuner/common.py                |      325 |       11 |     97% |80, 85, 90, 92, 134, 330, 353-354, 458, 504, 674 |
| amdsharktuner/amdsharktuner/constraint\_generator.py |      191 |        7 |     96% |376, 470, 472, 474, 589, 659, 668 |
| amdsharktuner/amdsharktuner/dispatch\_constraints.py |      288 |       13 |     95% |106-109, 149-158, 277, 380, 564, 634, 649 |
| amdsharktuner/amdsharktuner/dispatch\_parser.py      |      218 |        5 |     98% |36-38, 193, 480 |
| amdsharktuner/amdsharktuner/libtuner.py              |      675 |      405 |     40% |105-107, 110, 113, 116, 119, 122, 127-130, 134, 138, 142, 153, 161, 171, 219, 223-232, 322-453, 457-512, 523-545, 549-564, 568-610, 614-708, 755-810, 818-822, 852-858, 867-1001, 1043-1075, 1084-1119, 1172, 1281-1365, 1378-1511 |
| amdsharktuner/amdsharktuner/merge\_td\_specs.py      |       32 |       32 |      0% |     19-79 |
| amdsharktuner/amdsharktuner/process\_utils.py        |       89 |       32 |     64% |104-114, 144-183 |
| amdsharktuner/amdsharktuner/spec\_builder.py         |      127 |       23 |     82% |30-39, 76, 303-304, 314-342 |
| amdsharktuner/amdsharktuner/test\_utils.py           |       15 |        2 |     87% |     28-29 |
| amdsharktuner/boo\_tuner/\_\_init\_\_.py             |        0 |        0 |    100% |           |
| amdsharktuner/boo\_tuner/\_\_main\_\_.py             |        2 |        2 |      0% |       7-9 |
| amdsharktuner/boo\_tuner/boo\_tuner.py               |      208 |      139 |     33% |50-55, 59, 63, 67, 71, 75, 80, 89-131, 196-243, 256-371, 378-420, 424 |
| amdsharktuner/dispatch\_tuner/\_\_init\_\_.py        |        0 |        0 |    100% |           |
| amdsharktuner/dispatch\_tuner/\_\_main\_\_.py        |        2 |        2 |      0% |       7-9 |
| amdsharktuner/dispatch\_tuner/dispatch\_tuner.py     |       91 |       91 |      0% |     7-170 |
| amdsharktuner/model\_tuner/\_\_init\_\_.py           |        0 |        0 |    100% |           |
| amdsharktuner/model\_tuner/\_\_main\_\_.py           |        2 |        2 |      0% |       7-9 |
| amdsharktuner/model\_tuner/model\_tuner.py           |      127 |      127 |      0% |     7-246 |
| amdsharktuner/setup.py                               |       17 |       17 |      0% |      7-33 |
| amdsharktuner/tests/boo\_tuner\_test.py              |       72 |        0 |    100% |           |
| amdsharktuner/tests/candidate\_gen\_test.py          |      118 |        0 |    100% |           |
| amdsharktuner/tests/candidate\_ordering\_test.py     |       53 |        0 |    100% |           |
| amdsharktuner/tests/common\_test.py                  |      304 |        1 |     99% |       292 |
| amdsharktuner/tests/constraint\_generator\_test.py   |      287 |        0 |    100% |           |
| amdsharktuner/tests/dispatch\_constraints\_test.py   |      165 |        0 |    100% |           |
| amdsharktuner/tests/dispatch\_parser\_test.py        |      182 |        0 |    100% |           |
| amdsharktuner/tests/libtuner\_test.py                |      181 |        0 |    100% |           |
| amdsharktuner/tests/process\_utils\_test.py          |       20 |        1 |     95% |        20 |
| amdsharktuner/tests/spec\_builder\_test.py           |      135 |        0 |    100% |           |
|                                            **TOTAL** | **4162** |  **969** | **77%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/nod-ai/amd-shark-ai/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/nod-ai/amd-shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/nod-ai/amd-shark-ai/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/nod-ai/amd-shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fnod-ai%2Famd-shark-ai%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/nod-ai/amd-shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.