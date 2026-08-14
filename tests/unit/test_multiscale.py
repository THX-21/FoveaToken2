import unittest

from PIL import Image

from tokenfovea.multiscale import NativeProcessorProxy, aligned_native_plan, resize_native_image


class NativeImagePlanTest(unittest.TestCase):
    def test_plan_is_aligned_for_all_four_scales(self):
        image = Image.new("RGB", (913, 507))
        plan = aligned_native_plan(image, 28, 200704, 1605632)

        self.assertEqual(plan.grid_width % 8, 0)
        self.assertEqual(plan.grid_height % 8, 0)
        for divisor in (1, 2, 4, 8):
            resized = resize_native_image(image, plan, divisor)
            self.assertEqual(resized.size, (plan.width // divisor, plan.height // divisor))

    def test_processor_proxy_builds_main_and_auxiliary_inputs(self):
        class Batch(dict):
            pass

        class Processor:
            marker = "processor-attribute"

            def __call__(self, *args, **kwargs):
                images = kwargs["images"]
                self.assert_no_resize(kwargs)
                return Batch(image_sizes=[image.size for image in images])

            @staticmethod
            def assert_no_resize(kwargs):
                if kwargs["do_resize"] is not False:
                    raise AssertionError("processor resize must be disabled")

        proxy = NativeProcessorProxy(
            Processor(),
            pixel_per_token=32,
            min_pixels=32 * 32,
            max_pixels=256 * 256,
        )
        result = proxy(text=["prompt"], images=[Image.new("RGB", (101, 77))], videos=None)

        self.assertEqual(proxy.marker, "processor-attribute")
        main_width, main_height = result["image_sizes"][0]
        self.assertEqual(main_width % (32 * 8), 0)
        self.assertEqual(main_height % (32 * 8), 0)
        pending = result["_tokenfovea_native_inputs"]
        self.assertEqual(pending[4]["image_sizes"], [(main_width // 2, main_height // 2)])
        self.assertEqual(pending[16]["image_sizes"], [(main_width // 4, main_height // 4)])
        self.assertEqual(pending[64]["image_sizes"], [(main_width // 8, main_height // 8)])


if __name__ == "__main__":
    unittest.main()
