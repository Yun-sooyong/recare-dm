import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    # 팀 합의: Encounter -> PredictionData 모델명 변경(테이블 RENAME, 데이터 보존)
    # + 예측 결과 컬럼 3개 추가. autodetector 가 필드 추가 때문에 rename 으로
    # 인식 못 해 손으로 RenameModel 명시함.

    dependencies = [
        ("patients", "0003_encounter_readmitted"),
    ]

    operations = [
        migrations.RenameModel(
            old_name="Encounter",
            new_name="PredictionData",
        ),
        migrations.AlterField(
            model_name="predictiondata",
            name="patient",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="predictions",
                to="patients.patient",
            ),
        ),
        migrations.AddField(
            model_name="predictiondata",
            name="readmit_proba",
            field=models.FloatField(
                blank=True, null=True,
                help_text="30일 재입원 위험도 (0~1). 미예측 시 null",
            ),
        ),
        migrations.AddField(
            model_name="predictiondata",
            name="risk_level",
            field=models.CharField(
                blank=True, max_length=10,
                help_text="위험 등급 (high/medium/low 등). 미예측 시 빈 문자열",
            ),
        ),
        migrations.AddField(
            model_name="predictiondata",
            name="predicted_at",
            field=models.DateTimeField(
                blank=True, null=True,
                help_text="예측 수행 시각. 미예측 시 null",
            ),
        ),
    ]
