# Tests

This project uses unit and integration tests with Python files and notebooks.

 * In the unit tests we just make sure the notebook runs.
 * In the integration tests we use a bigger dataset for more epochs and we test that the metrics are what we expect.

For more information, see a [quick introduction to unit and integration tests](https://miguelgfierro.com/blog/2018/a-beginners-guide-to-python-testing/). To manually execute the unit tests in the different environments, first **make sure you are in the correct environment as described in the [README.md](../README.md)**

Tests are automatically run as part of a DevOps pipeline. The pipelines are defined in the `.yml` files in [tests/ci](./ci) with filenames that align with pipeline names.

## Test execution

**Click on the following menus** to see more details on how to execute the unit and integration tests:

<details>
<summary><strong><em>Unit tests (click to expand)</em></strong></summary>

Unit tests ensure that each class or function behaves as it should. Every time a developer makes a pull request to staging or master branch, a battery of unit tests is executed.

**Note that the next instructions execute the tests from the root folder.**



</details>



<details>
<summary><strong><em>Integration tests (click to expand)</em></strong></summary>

Integration tests make sure that the program results are acceptable

**Note that the next instructions execute the tests from the root folder.**


</details>


## How to create tests on notebooks with Papermill

In the notebooks of these repo we use [Papermill](https://github.com/nteract/papermill) in unit and integration tests.

In the unit tests we just make sure the notebook runs. In the integration tests, we use a bigger dataset for more epochs and we test that the metrics are what we expect.

For a deep overview on how to integrate papermill on unit and integration test, please refer to [this guide from Microsoft Recommenders repo](https://github.com/microsoft/recommenders/blob/master/tests/README.md#how-to-create-tests-on-notebooks-with-papermill).

More details on how to integrate Papermill with notebooks can be found in their [repo](https://github.com/nteract/papermill).