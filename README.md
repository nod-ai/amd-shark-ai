# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/nod-ai/amd-shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                             |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| amdsharktuner/amdsharktuner/\_\_init\_\_.py                      |        7 |        0 |    100% |           |
| amdsharktuner/amdsharktuner/candidate\_gen.py                    |       62 |       26 |     58% |47, 79-80, 103-119, 135-147, 157-162, 168-183 |
| amdsharktuner/amdsharktuner/candidate\_ordering.py               |      103 |       11 |     89% |92-95, 115-116, 190-198 |
| amdsharktuner/amdsharktuner/common.py                            |      225 |       10 |     96% |115, 120, 125, 127, 169, 317, 340-341, 414, 460 |
| amdsharktuner/amdsharktuner/constraint\_generator.py             |       85 |        7 |     92% |246, 255-265 |
| amdsharktuner/amdsharktuner/dispatch\_parser.py                  |      154 |        8 |     95% |42-44, 145, 296, 315-318, 448 |
| amdsharktuner/amdsharktuner/libtuner.py                          |      620 |      361 |     42% |104-106, 109, 112, 115, 118, 121, 133, 137, 141, 152, 160, 170, 203, 207-216, 308, 457-458, 463-518, 529-551, 555-569, 573-614, 618-712, 720-724, 754-760, 769-945, 987-1019, 1028-1063, 1116, 1225-1309, 1322-1455 |
| amdsharktuner/amdsharktuner/merge\_td\_specs.py                  |       32 |       32 |      0% |     19-79 |
| amdsharktuner/amdsharktuner/process\_utils.py                    |       84 |       32 |     62% |104-114, 144-183 |
| amdsharktuner/amdsharktuner/rocm/\_\_init\_\_.py                 |        4 |        0 |    100% |           |
| amdsharktuner/amdsharktuner/rocm/rocm\_candidate\_ordering.py    |        7 |        0 |    100% |           |
| amdsharktuner/amdsharktuner/rocm/rocm\_common.py                 |      152 |       27 |     82% |264, 380-435 |
| amdsharktuner/amdsharktuner/rocm/rocm\_constraint\_generators.py |       61 |        9 |     85% |66, 75-81, 254-257, 266-269, 313, 321 |
| amdsharktuner/amdsharktuner/rocm/rocm\_dispatch\_constraints.py  |      282 |       13 |     95% |111, 181-190, 299, 391, 577, 647, 666, 819, 822 |
| amdsharktuner/amdsharktuner/rocm/rocm\_parsers.py                |       61 |        0 |    100% |           |
| amdsharktuner/amdsharktuner/rocm/rocm\_solutions.py              |      174 |        7 |     96% |151, 400-401, 451, 547, 549, 551 |
| amdsharktuner/amdsharktuner/rocm/rocm\_tuners.py                 |      120 |       44 |     63% |31-32, 42, 45, 58, 64, 71, 75-92, 95, 103-104, 108, 114, 129, 141, 154, 160, 167, 171-176, 179, 187-188, 192, 198, 205, 212, 222-223, 227, 233 |
| amdsharktuner/amdsharktuner/spec\_builder.py                     |      130 |       24 |     82% |28, 33-42, 79, 306-307, 317-345 |
| amdsharktuner/amdsharktuner/test\_utils.py                       |       15 |        2 |     87% |     28-29 |
| amdsharktuner/amdsharktuner/tuner\_base.py                       |       22 |        5 |     77% |20, 34, 39, 45, 56 |
| amdsharktuner/boo\_tuner/\_\_init\_\_.py                         |        0 |        0 |    100% |           |
| amdsharktuner/boo\_tuner/\_\_main\_\_.py                         |        2 |        2 |      0% |       7-9 |
| amdsharktuner/boo\_tuner/boo\_tuner.py                           |      209 |      140 |     33% |50-55, 59, 63, 67, 71, 75, 80, 89-131, 196-243, 256-374, 381-425, 429 |
| amdsharktuner/dispatch\_tuner/\_\_init\_\_.py                    |        0 |        0 |    100% |           |
| amdsharktuner/dispatch\_tuner/\_\_main\_\_.py                    |        2 |        2 |      0% |       7-9 |
| amdsharktuner/dispatch\_tuner/dispatch\_tuner.py                 |       91 |       91 |      0% |     7-170 |
| amdsharktuner/fusilli\_tuner/\_\_init\_\_.py                     |        2 |        0 |    100% |           |
| amdsharktuner/fusilli\_tuner/\_\_main\_\_.py                     |        2 |        2 |      0% |       7-9 |
| amdsharktuner/fusilli\_tuner/fusilli\_tuner.py                   |      240 |      103 |     57% |309-310, 324-375, 390-485, 493, 498-531, 535 |
| amdsharktuner/model\_tuner/\_\_init\_\_.py                       |        0 |        0 |    100% |           |
| amdsharktuner/model\_tuner/\_\_main\_\_.py                       |        2 |        2 |      0% |       7-9 |
| amdsharktuner/model\_tuner/model\_tuner.py                       |      127 |      127 |      0% |     7-246 |
| amdsharktuner/setup.py                                           |       17 |       17 |      0% |      7-33 |
| amdsharktuner/tests/\_\_init\_\_.py                              |        0 |        0 |    100% |           |
| amdsharktuner/tests/boo\_tuner\_test.py                          |       72 |        0 |    100% |           |
| amdsharktuner/tests/candidate\_gen\_test.py                      |      129 |        0 |    100% |           |
| amdsharktuner/tests/candidate\_ordering\_test.py                 |       20 |        0 |    100% |           |
| amdsharktuner/tests/common\_test.py                              |      260 |        1 |     99% |       249 |
| amdsharktuner/tests/conftest.py                                  |        7 |        2 |     71% |     16-18 |
| amdsharktuner/tests/constraint\_generator\_test.py               |      168 |        0 |    100% |           |
| amdsharktuner/tests/dispatch\_parser\_test.py                    |      149 |        0 |    100% |           |
| amdsharktuner/tests/fusilli\_tuner\_test.py                      |      167 |        0 |    100% |           |
| amdsharktuner/tests/libtuner\_test.py                            |      161 |        0 |    100% |           |
| amdsharktuner/tests/process\_utils\_test.py                      |       19 |        1 |     95% |        20 |
| amdsharktuner/tests/rocm/\_\_init\_\_.py                         |        0 |        0 |    100% |           |
| amdsharktuner/tests/rocm/rocm\_candidate\_ordering\_test.py      |       52 |        0 |    100% |           |
| amdsharktuner/tests/rocm/rocm\_common\_test.py                   |      125 |        0 |    100% |           |
| amdsharktuner/tests/rocm/rocm\_constraint\_generator\_test.py    |      460 |        0 |    100% |           |
| amdsharktuner/tests/rocm/rocm\_dispatch\_constraints\_test.py    |      202 |        0 |    100% |           |
| amdsharktuner/tests/rocm/rocm\_dispatch\_parser\_test.py         |       27 |        0 |    100% |           |
| amdsharktuner/tests/rocm/rocm\_parsers\_test.py                  |       59 |        0 |    100% |           |
| amdsharktuner/tests/spec\_builder\_test.py                       |      135 |        0 |    100% |           |
| **TOTAL**                                                        | **5306** | **1108** | **79%** |           |


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