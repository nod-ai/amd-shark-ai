# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/nod-ai/amd-shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                             |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| amdsharktuner/amdsharktuner/\_\_init\_\_.py                      |        7 |        0 |    100% |           |
| amdsharktuner/amdsharktuner/candidate\_gen.py                    |       62 |       26 |     58% |47, 79-81, 101-106, 121-133, 143-148, 154-170 |
| amdsharktuner/amdsharktuner/candidate\_ordering.py               |      106 |       11 |     90% |107-110, 130-131, 205-213 |
| amdsharktuner/amdsharktuner/common.py                            |      222 |       10 |     95% |80, 85, 90, 92, 134, 269, 292-293, 367, 413 |
| amdsharktuner/amdsharktuner/constraint\_generator.py             |       85 |        7 |     92% |246, 255-265 |
| amdsharktuner/amdsharktuner/dispatch\_parser.py                  |      178 |        5 |     97% |36-38, 193, 480 |
| amdsharktuner/amdsharktuner/libtuner.py                          |      667 |      414 |     38% |105-107, 110, 113, 116, 119, 122, 127-130, 134, 138, 142, 153, 161, 171, 219, 223-232, 322-453, 457-512, 523-545, 549-564, 568-610, 614-708, 755-810, 818-822, 852-858, 867-1020, 1062-1094, 1103-1138, 1191, 1300-1384, 1397-1530 |
| amdsharktuner/amdsharktuner/merge\_td\_specs.py                  |       32 |       32 |      0% |     19-79 |
| amdsharktuner/amdsharktuner/process\_utils.py                    |       84 |       32 |     62% |104-114, 144-183 |
| amdsharktuner/amdsharktuner/rocm/\_\_init\_\_.py                 |        3 |        0 |    100% |           |
| amdsharktuner/amdsharktuner/rocm/rocm\_common.py                 |       75 |        1 |     99% |       211 |
| amdsharktuner/amdsharktuner/rocm/rocm\_constraint\_generators.py |       29 |        4 |     86% |66, 76, 182, 190 |
| amdsharktuner/amdsharktuner/rocm/rocm\_dispatch\_constraints.py  |      278 |       11 |     96% |111, 181-190, 299, 391, 577, 647, 662 |
| amdsharktuner/amdsharktuner/rocm/rocm\_solutions.py              |      170 |        4 |     98% |421, 517, 519, 521 |
| amdsharktuner/amdsharktuner/rocm/rocm\_tuners.py                 |      120 |       44 |     63% |31-32, 42, 45, 58, 64, 71, 75-92, 95, 103-104, 108, 114, 129, 141, 154, 160, 167, 171-177, 187, 195-196, 200, 206, 213, 220, 230-231, 235, 241 |
| amdsharktuner/amdsharktuner/spec\_builder.py                     |      127 |       23 |     82% |30-39, 76, 303-304, 314-342 |
| amdsharktuner/amdsharktuner/test\_utils.py                       |       15 |        2 |     87% |     28-29 |
| amdsharktuner/amdsharktuner/tuner\_base.py                       |       22 |        5 |     77% |20, 34, 39, 45, 56 |
| amdsharktuner/boo\_tuner/\_\_init\_\_.py                         |        0 |        0 |    100% |           |
| amdsharktuner/boo\_tuner/\_\_main\_\_.py                         |        2 |        2 |      0% |       7-9 |
| amdsharktuner/boo\_tuner/boo\_tuner.py                           |      208 |      139 |     33% |50-55, 59, 63, 67, 71, 75, 80, 89-131, 196-243, 256-371, 378-420, 424 |
| amdsharktuner/dispatch\_tuner/\_\_init\_\_.py                    |        0 |        0 |    100% |           |
| amdsharktuner/dispatch\_tuner/\_\_main\_\_.py                    |        2 |        2 |      0% |       7-9 |
| amdsharktuner/dispatch\_tuner/dispatch\_tuner.py                 |       91 |       91 |      0% |     7-170 |
| amdsharktuner/model\_tuner/\_\_init\_\_.py                       |        0 |        0 |    100% |           |
| amdsharktuner/model\_tuner/\_\_main\_\_.py                       |        2 |        2 |      0% |       7-9 |
| amdsharktuner/model\_tuner/model\_tuner.py                       |      127 |      127 |      0% |     7-246 |
| amdsharktuner/setup.py                                           |       17 |       17 |      0% |      7-33 |
| amdsharktuner/tests/boo\_tuner\_test.py                          |       72 |        0 |    100% |           |
| amdsharktuner/tests/candidate\_gen\_test.py                      |      129 |        0 |    100% |           |
| amdsharktuner/tests/candidate\_ordering\_test.py                 |       64 |        0 |    100% |           |
| amdsharktuner/tests/common\_test.py                              |      217 |        1 |     99% |       220 |
| amdsharktuner/tests/conftest.py                                  |        7 |        2 |     71% |     16-18 |
| amdsharktuner/tests/constraint\_generator\_test.py               |      341 |        0 |    100% |           |
| amdsharktuner/tests/dispatch\_parser\_test.py                    |      183 |        0 |    100% |           |
| amdsharktuner/tests/libtuner\_test.py                            |      181 |        0 |    100% |           |
| amdsharktuner/tests/process\_utils\_test.py                      |       19 |        1 |     95% |        20 |
| amdsharktuner/tests/rocm/\_\_init\_\_.py                         |        0 |        0 |    100% |           |
| amdsharktuner/tests/rocm/rocm\_common\_test.py                   |       98 |        0 |    100% |           |
| amdsharktuner/tests/rocm/rocm\_dispatch\_constraints\_test.py    |      178 |        0 |    100% |           |
| amdsharktuner/tests/spec\_builder\_test.py                       |      135 |        0 |    100% |           |
| **TOTAL**                                                        | **4355** | **1015** | **77%** |           |


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