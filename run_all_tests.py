"""
综合测试套件：运行所有测试用例
"""
import sys
import subprocess
from pathlib import Path


def run_test(test_file: str, description: str) -> bool:
    """运行单个测试文件"""
    print("\n" + "="*70)
    print(f"运行: {description}")
    print("="*70)

    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(result.stdout)
        print(f"✅ {description} - 通过")
        return True
    else:
        print(result.stdout)
        print(result.stderr)
        print(f"❌ {description} - 失败")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("水网优化模型 - 综合测试套件")
    print("="*70)

    tests = [
        ("test_core_functionality.py", "核心功能测试"),
        ("example_water_network_optimization.py", "完整优化示例"),
    ]

    results = []
    for test_file, description in tests:
        test_path = Path(__file__).parent / test_file
        if test_path.exists():
            passed = run_test(str(test_path), description)
            results.append((description, passed))
        else:
            print(f"⚠️  跳过: {test_file} (文件不存在)")
            results.append((description, None))

    # 汇总报告
    print("\n" + "="*70)
    print("测试汇总报告")
    print("="*70)

    passed_count = sum(1 for _, result in results if result is True)
    failed_count = sum(1 for _, result in results if result is False)
    skipped_count = sum(1 for _, result in results if result is None)

    for description, result in results:
        if result is True:
            status = "✅ 通过"
        elif result is False:
            status = "❌ 失败"
        else:
            status = "⚠️  跳过"
        print(f"{status} - {description}")

    print(f"\n总计: {len(results)} 个测试")
    print(f"  通过: {passed_count}")
    print(f"  失败: {failed_count}")
    print(f"  跳过: {skipped_count}")

    if failed_count == 0 and passed_count > 0:
        print("\n🎉 所有测试通过!")
        return True
    elif failed_count > 0:
        print("\n⚠️  存在失败的测试")
        return False
    else:
        print("\n⚠️  没有运行任何测试")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
