import sys
import json
import os
import time
from datetime import datetime
from tqdm import tqdm

def update_progress(progress):
    """진행률을 출력하는 함수"""
    print(json.dumps({"progress": progress}), flush=True)  # JSON 형식으로 출력

def analyze_video(video_folder):
    try:
        # 전체 작업 단계 정의
        total_steps = 100
        progress_bar = tqdm(total=total_steps, desc="분석 진행중", unit="%")
        
        # 초기 진행률
        update_progress(0)
        
        results_dir = os.path.join('data', 'analysis_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # 데이터 준비 단계 (20%)
        for i in range(20):
            time.sleep(0.1)  # 실제 작업으로 대체
            progress_bar.update(1)
            update_progress(progress_bar.n)
        
        # 현재 시간을 파일명으로 사용
        now = datetime.now()
        filename = now.strftime('%Y%m%d_%H%M%S.json')
        
        # 분석 중간 단계 (30%)
        for i in range(30):
            time.sleep(0.1)  # 실제 작업으로 대체
            progress_bar.update(1)
            update_progress(progress_bar.n)
        
        # 분석 결과 생성
        analysis_result = {
            "summary": {
                "감정 토탈": {
                    "positive": 9.07,
                    "negative": 70.49,
                    "neutral": 20.44
                },
                "그룹별 감정 분석": [
                    {
                        "group": "섭취 1번째",
                        "group_number": 7,
                        "positive": 15.31,
                        "negative": 77.18,
                        "neutral": 7.51
                    },
                    {
                        "group": "섭취 2번째",
                        "group_number": 8,
                        "positive": 17.42,
                        "negative": 50.0,
                        "neutral": 32.58
                    },
                    {
                        "group": "섭취 3번째",
                        "group_number": 0,
                        "positive": 2.79,
                        "negative": 62.69,
                        "neutral": 34.53
                    },
                    {
                        "group": "섭취 4번째",
                        "group_number": 1,
                        "positive": 0.05,
                        "negative": 52.03,
                        "neutral": 47.92
                    },
                    {
                        "group": "섭취 5번째",
                        "group_number": 2,
                        "positive": 6.81,
                        "negative": 79.01,
                        "neutral": 14.18
                    },
                    {
                        "group": "섭취 6번째",
                        "group_number": 3,
                        "positive": 3.58,
                        "negative": 80.57,
                        "neutral": 15.84
                    },
                    {
                        "group": "섭취 7번째",
                        "group_number": 4,
                        "positive": 9.77,
                        "negative": 85.41,
                        "neutral": 4.81
                    },
                    {
                        "group": "섭취 8번째",
                        "group_number": 5,
                        "positive": 14.51,
                        "negative": 59.71,
                        "neutral": 25.78
                    },
                    {
                        "group": "섭취 9번째",
                        "group_number": 6,
                        "positive": 11.41,
                        "negative": 87.79,
                        "neutral": 0.8
                    }
                ],
                "행동 토탈": {
                    "positive_behavior": 49.54,
                    "negative_behavior": 50.46
                },
                "프레임별 행동 분석": [
                    {
                        "Start Time (s)": 0,
                        "Positive (%)": 80.0,
                        "Negative (%)": 20.0
                    },
                    {
                        "Start Time (s)": 8,
                        "Positive (%)": 78.57142857142857,
                        "Negative (%)": 21.428571428571427
                    },
                    {
                        "Start Time (s)": 34,
                        "Positive (%)": 100.0,
                        "Negative (%)": 0.0
                    },
                    {
                        "Start Time (s)": 35,
                        "Positive (%)": 50.0,
                        "Negative (%)": 50.0
                    },
                    {
                        "Start Time (s)": 36,
                        "Positive (%)": 0.0,
                        "Negative (%)": 100.0
                    },
                    {
                        "Start Time (s)": 57,
                        "Positive (%)": 28.57142857142857,
                        "Negative (%)": 71.42857142857143
                    },
                    {
                        "Start Time (s)": 71,
                        "Positive (%)": 20.0,
                        "Negative (%)": 80.0
                    },
                    {
                        "Start Time (s)": 139,
                        "Positive (%)": 30.0,
                        "Negative (%)": 70.0
                    },
                    {
                        "Start Time (s)": 140,
                        "Positive (%)": 33.33333333333333,
                        "Negative (%)": 66.66666666666666
                    },
                    {
                        "Start Time (s)": 150,
                        "Positive (%)": 58.82352941176471,
                        "Negative (%)": 41.17647058823529
                    },
                    {
                        "Start Time (s)": 169,
                        "Positive (%)": 37.25490196078432,
                        "Negative (%)": 62.74509803921568
                    },
                    {
                        "Start Time (s)": 187,
                        "Positive (%)": 50.0,
                        "Negative (%)": 50.0
                    },
                    {
                        "Start Time (s)": 191,
                        "Positive (%)": 100.0,
                        "Negative (%)": 0.0
                    },
                    {
                        "Start Time (s)": 192,
                        "Positive (%)": 50.0,
                        "Negative (%)": 50.0
                    },
                    {
                        "Start Time (s)": 194,
                        "Positive (%)": 50.0,
                        "Negative (%)": 50.0
                    },
                    {
                        "Start Time (s)": 195,
                        "Positive (%)": 0.0,
                        "Negative (%)": 100.0
                    },
                    {
                        "Start Time (s)": 211,
                        "Positive (%)": 0.0,
                        "Negative (%)": 100.0
                    },
                    {
                        "Start Time (s)": 212,
                        "Positive (%)": 16.666666666666664,
                        "Negative (%)": 83.33333333333334
                    },
                    {
                        "Start Time (s)": 214,
                        "Positive (%)": 0.0,
                        "Negative (%)": 100.0
                    },
                    {
                        "Start Time (s)": 215,
                        "Positive (%)": 0.0,
                        "Negative (%)": 100.0
                    },
                    {
                        "Start Time (s)": 260,
                        "Positive (%)": 66.66666666666666,
                        "Negative (%)": 33.33333333333333
                    },
                    {
                        "Start Time (s)": 286,
                        "Positive (%)": 50.0,
                        "Negative (%)": 50.0
                    },
                    {
                        "Start Time (s)": 313,
                        "Positive (%)": 79.16666666666666,
                        "Negative (%)": 20.833333333333336
                    },
                    {
                        "Start Time (s)": 315,
                        "Positive (%)": 51.42857142857142,
                        "Negative (%)": 48.57142857142857
                    },
                    {
                        "Start Time (s)": 419,
                        "Positive (%)": 88.46153846153847,
                        "Negative (%)": 11.53846153846154
                    },
                    {
                        "Start Time (s)": 420,
                        "Positive (%)": 85.96491228070175,
                        "Negative (%)": 14.035087719298245
                    },
                    {
                        "Start Time (s)": 421,
                        "Positive (%)": 91.11111111111111,
                        "Negative (%)": 8.88888888888889
                    },
                    {
                        "Start Time (s)": 446,
                        "Positive (%)": 75.0,
                        "Negative (%)": 25.0
                    },
                    {
                        "Start Time (s)": 447,
                        "Positive (%)": 51.35135135135135,
                        "Negative (%)": 48.64864864864865
                    },
                    {
                        "Start Time (s)": 448,
                        "Positive (%)": 57.14285714285714,
                        "Negative (%)": 42.85714285714285
                    },
                    {
                        "Start Time (s)": 453,
                        "Positive (%)": 42.10526315789473,
                        "Negative (%)": 57.89473684210527
                    },
                    {
                        "Start Time (s)": 458,
                        "Positive (%)": 60.0,
                        "Negative (%)": 40.0
                    },
                    {
                        "Start Time (s)": 479,
                        "Positive (%)": 48.14814814814815,
                        "Negative (%)": 51.85185185185185
                    },
                    {
                        "Start Time (s)": 480,
                        "Positive (%)": 54.54545454545454,
                        "Negative (%)": 45.45454545454545
                    }
                ],
                "총합 감정+행동": {
                    "Y_positive": 17.16,
                    "Y_negative": 66.48,
                    "Y_neutral": 16.35
                }
            },
            "analysis": {
                "total_analysis": "안녕",
                "emotion_analysis": "햄버거를 소비하는 과정에서 고객들의 감정 변화를 분석하였습니다. 그룹별로 살펴보니, 각 섭취 횟수별로 다양한 감정 변화 패턴을 보였습니다.\n\n1번째 섭취 그룹에서는 <b>긍정적 감정</b>이 <b>15.31%</b>로 나타났으며, <b>부정적 감정</b>이 <b>77.18%</b>로 대부분을 차지했습니다. <b>중립적 감정</b>은 <b>7.51%</b>로 나타났습니다.\n\n2번째 섭취 그룹에서는 <b>긍정적 감정</b>이 <b>17.42%</b>, <b>부정적 감정</b>이 <b>50.0%</b>, <b>중립적 감정</b>이 <b>32.58%</b>로 나타났습니다. 이 그룹에서는 <b>부정적 감정</b>이 감소하고 <b>중립적 감정</b>이 증가한 것을 확인할 수 있습니다.\n\n3번째 섭취 그룹에서는 <b>긍정적 감정</b>이 <b>2.79%</b>, <b>부정적 감정</b>이 <b>62.69%</b>, <b>중립적 감정</b>이 <b>34.53%</b>로 나타났습니다. 이 그룹에서는 <b>긍정적 감정</b>이 크게 감소하였습니다.\n\n그 이후의 섭취 그룹들에서도 감정 변화가 다양하게 나타났습니다. 특히, 9번째 섭취 그룹에서는 <b>부정적 감정</b>이 <b>87.79%</b>로 가장 높게 나타났으며, <b>긍정적 감정</b>은 <b>11.41%</b>, <b>중립적 감정</b>은 <b>0.8%</b>로 나타났습니다.\n\n이러한 분석을 통해 고객들이 햄버거를 섭취하는 과정에서 감정이 어떻게 변화하는지 파악할 수 있습니다. 이를 바탕으로 고객의 감정 변화를 고려한 <b>판매 전략</b>을 수립할 수 있습니다. 예를 들어, 감정이 부정적으로 변화하는 시점에는 고객에게 긍정적인 경험을 제공할 수 있는 서비스를 제공하거나, 제품의 품질을 개선하는 등의 방법을 고려해 볼 수 있습니다.",
                "behavior_analysis": "햄버거 소비에 대한 행동 데이터를 분석하면, 전체적인 행동 패턴에서는 긍정적인 행동이 <b>49.54%</b>로, 부정적인 행동이 <b>50.46%</b>로 나타납니다. 이는 소비자들의 햄버거에 대한 반응이 상당히 균등하게 분포되어 있음을 보여줍니다.\n\n프레임별로 행동을 분석하면, 시작 시점에서 긍정 행동이 80%로 높게 나타나며, 이 후 점차적으로 감소하는 경향을 보입니다. 이는 햄버거를 처음 섭취하는 시점에서의 기대감이나 호기심 등이 긍정적인 행동을 유발했을 가능성이 있습니다.\n\n그러나 시간이 지남에 따라서 긍정적인 행동의 비율은 감소하고, 부정적인 행동의 비율이 상승하는 경향을 보입니다. 특히, 36초에서 100%의 부정적인 행동이 나타나는 등 몇몇 시점에서 부정적인 행동이 급증하는 모습을 볼 수 있습니다. 이러한 패턴은 햄버거의 품질이나 맛 등이 소비자의 기대치를 충족시키지 못했을 가능성을 시사합니다.\n\n그러나 최종 시점에서는 긍정적인 행동이 <b>54.55%</b>로 다시 상승하는 모습을 보입니다. 이는 햄버거의 후반 부분에서 다시금 소비자의 만족도가 상승하였음을 의미하며, 이는 햄버거의 특정 부분이나 요소가 소비자에게 긍정적인 반응을 이끌어냈을 가능성이 있습니다.\n\n따라서 이러한 행동 변화 패턴을 바탕으로 <b>판매 전략</b>을 개선하려면, 햄버거의 초기 품질이나 맛을 유지하면서도 중반부의 소비자 만족도를 향상시킬 수 있는 방안을 모색하는 것이 필요해 보입니다. 이를 통해 소비자의 전반적인 햄버거 섭취 경험을 긍정적으로 만들어 판매를 촉진할 수 있을 것입니다."
            }
        }
        
        # 결과 저장 전 단계 (30%)
        for i in range(30):
            time.sleep(0.1)  # 실제 작업으로 대체
            progress_bar.update(1)
            update_progress(progress_bar.n)
        
        # 결과를 JSON 파일로 저장
        output_file = os.path.join(results_dir, filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=4)
        
        # 완료
        progress_bar.update(20)  # 마지막 20% 업데이트
        update_progress(100)
        progress_bar.close()
        
        return {
            "status": "success",
            "results_file": output_file,
            "message": "분석이 성공적으로 완료되었습니다."
        }
        
    except Exception as e:
        if 'progress_bar' in locals():
            progress_bar.close()
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_folder = sys.argv[1]
        result = analyze_video(video_folder)
        print(json.dumps(result))
        sys.stdout.flush()