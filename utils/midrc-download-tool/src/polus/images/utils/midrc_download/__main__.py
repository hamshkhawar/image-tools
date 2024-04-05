"""Package entrypoint for the midrc_download package."""

# Base packages
import logging
from os import environ
from pathlib import Path
from typing import Optional

import polus.images.utils.midrc_download.midrc_download as md
import polus.images.utils.midrc_download.utils as ut
import typer

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.images.utils.midrc_download.midrc_download")
logger.setLevel(POLUS_LOG)

POLUS_IMG_EXT = environ.get("POLUS_IMG_EXT", ".ome.tif")

app = typer.Typer(help="Midrc Download.")


@app.command()
def main(  # noqa: PLR0913
    study_modality: Optional[list[str]] = typer.Option(
        None,
        "--studyModality",
        help="The modalities of the imaging study.",
    ),
    study_description: Optional[list[str]] = typer.Option(
        None,
        "--studyDescription",
        help="The description of the imaging study.",
    ),
    loinc_method: Optional[list[str]] = typer.Option(
        None,
        "--loincMethod",
        help="The LOINC method or imaging modality associated with LOINC code.",
    ),
    midrc_type: ut.MIDRCTYPES = typer.Option(
        ...,
        "--MidrcType",
        help="The node_id in the data model utilized in queries and API requests.",
    ),
    loinc_system: Optional[list[str]] = typer.Option(
        None,
        "--loincSystem",
        help="The LOINC system or body part examined associated with LOINC code.",
    ),
    loinc_long_common_name: Optional[list[str]] = typer.Option(
        None,
        "--loincLongCommonName",
        help="The LOINC system or body part examined associated with LOINC code.",
    ),
    days_from_study_to_neg_covid_test: Optional[list[int]] = typer.Option(
        None,
        "--daysFromStudyToNegCovidTest",
        help="The LOINC long common name.",
    ),
    study_year: Optional[list[str]] = typer.Option(
        None,
        "--studyYear",
        help="The year when imaging study was performed.",
    ),
    study_year_shifted: Optional[list[str]] = typer.Option(
        None,
        "--studyYearShifted",
        help="The year when imaging study was performed.",
    ),
    days_from_study_to_pos_covid_test: Optional[list[int]] = typer.Option(
        None,
        "--studyYearShifted",
        help="The year when imaging study was performed.",
    ),
    age_at_imaging: Optional[list[str]] = typer.Option(
        None,
        "--ageAtImaging",
        help="The age of the study participant.",
    ),
    loinc_contrast: Optional[list[str]] = typer.Option(
        None,
        "--loincContrast",
        help="The indicator if the image was completed with or without contrast",
    ),
    project_id: Optional[list[str]] = typer.Option(
        None,
        "--projectId",
        help="The code of the project that this dataset belongs.",
    ),
    body_part_examined: Optional[list[str]] = typer.Option(
        None,
        "--bodyPartExamined",
        help="Body Part Examined.",
    ),
    loinc_code: Optional[list[str]] = typer.Option(
        None,
        "--loincCode",
        help="The loinc code",
    ),
    sex: Optional[list[str]] = typer.Option(
        None,
        "--sex",
        help="A gender information.",
    ),
    race: Optional[list[str]] = typer.Option(
        None,
        "--race",
        help="Race.",
    ),
    age_at_index: Optional[list[str]] = typer.Option(
        None,
        "--ageAtIndex",
        help="The age of the study participant.",
    ),
    index_event: Optional[list[str]] = typer.Option(
        None,
        "--indexEvent",
        help="The age of the study participant.",
    ),
    covid19_positive: Optional[list[str]] = typer.Option(
        None,
        "--covid19Positive",
        help="An indicator of whether patient has covid infection or not.",
    ),
    ethnicity: Optional[list[str]] = typer.Option(
        None,
        "--ethnicity",
        help="A racial or cultural background.",
    ),
    data_format: Optional[list[str]] = typer.Option(
        None,
        "--dataFormat",
        help="The file format, physical medium, or dimensions of the resource.",
    ),
    data_type: Optional[list[str]] = typer.Option(
        None,
        "--dataType",
        help="The file format, physical medium, or dimensions of the resource.",
    ),
    data_category: Optional[list[str]] = typer.Option(
        None,
        "--dataCategory",
        help="Image files and metadata related to several imaging series.",
    ),
    data_file_annotation_name: Optional[list[str]] = typer.Option(
        None,
        "--dataFileAnnotationName",
        help="Image files and metadata related to several imaging series.",
    ),
    data_file_annotation_method: Optional[list[str]] = typer.Option(
        None,
        "--dataFileAnnotationMethod",
        help="Image files and metadata related to several imaging series.",
    ),
    midrc_mRALE_score: Optional[list[str]] = typer.Option(
        None,
        "--midrcMRALEScore",
        help="Image files and metadata related to several imaging series.",
    ),
    test_days_from_index: Optional[list[str]] = typer.Option(
        None,
        "--testDaysFromIndex",
        help="Image files and metadata related to several imaging series.",
    ),
    test_method: Optional[list[str]] = typer.Option(
        None,
        "--testMethod",
        help="Image files and metadata related to several imaging series.",
    ),
    test_name: Optional[list[str]] = typer.Option(
        None,
        "--testName",
        help="Image files and metadata related to several imaging series.",
    ),
    test_result_text: Optional[list[str]] = typer.Option(
        None,
        "--testResultText",
        help="Image files and metadata related to several imaging series.",
    ),
    condition_name: Optional[list[str]] = typer.Option(
        None,
        "--conditionName",
        help="Image files and metadata related to several imaging series.",
    ),
    condition_code_system: Optional[list[str]] = typer.Option(
        None,
        "--conditionCodeSystem",
        help="Image files and metadata related to several imaging series.",
        ),
    condition_code: Optional[list[str]] = typer.Option(
        None,
        "--conditionCode",
        help="Image files and metadata related to several imaging series.",
        ),
    days_to_condition_start: Optional[list[str]] = typer.Option(
        None,
        "--daysToConditionStart",
        help="Image files and metadata related to several imaging series.",
        ),
    days_to_condition_end: Optional[list[str]] = typer.Option(
        None,
        "--daysToConditionEnd",
        help="Image files and metadata related to several imaging series.",
        ),
    days_to_medication_start: Optional[list[str]] = typer.Option(
        None,
        "--daysToMedicationStart",
        help="Image files and metadata related to several imaging series.",
        ),
    dose_sequence_number: Optional[list[str]] = typer.Option(
        None,
        "--doseSequenceNumber",
        help="Image files and metadata related to several imaging series.",
        ),
    medication_code: Optional[list[str]] = typer.Option(
        None,
        "--medicationCode",
        help="Image files and metadata related to several imaging series.",
        ),
    medication_manufacturer: Optional[list[str]] = typer.Option(
        None,
        "--medicationManufacturer",
        help="Image files and metadata related to several imaging series.",
        ),
    medication_name: Optional[list[str]] = typer.Option(
        None,
        "--medicationName",
        help="Image files and metadata related to several imaging series.",
        ),
    medication_type: Optional[list[str]] = typer.Option(
        None,
        "--medicationType",
        help="Image files and metadata related to several imaging series.",
        ),
    annotation_method: Optional[list[str]] = typer.Option(
        None,
        "--annotationMethod",
        help="Image files and metadata related to several imaging series.",
        ),
    annotation_id: Optional[list[str]] = typer.Option(
        None,
        "--annotationMethod",
        help="Image files and metadata related to several imaging series.",
        ),
    breathing_support_type: Optional[list[str]] = typer.Option(
        None,
        "--breathingSupportType",
        help="Image files and metadata related to several imaging series.",
        ),
    source_node: Optional[list[str]] = typer.Option(
        None,
        "--sourceNode",
        help="A package of image files and metadata related to several imaging series.",
    ),
    first: Optional[int] = typer.Option(
        None,
        "--first",
        help="Number of rows to return.",
    ),
    offset: Optional[int] = typer.Option(
        None,
        "--offset",
        help="Starting position.",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--outDir",
        help="Output directory.",
        exists=True,
        writable=True,
        file_okay=False,
        resolve_path=True,
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-v",
        help="Preview of expected outputs (dry-run)",
        show_default=False
    )
    ) -> None:
    """Midrc Download."""

    logger.info(f"studyModality: {study_modality}")
    logger.info(f"studyDescription: {study_description}")
    logger.info(f"loincMethod: {loinc_method}")
    logger.info(f"MidrcType: {midrc_type}")
    logger.info(f"loincSystem: {loinc_system}")
    logger.info(f"loincLongCommonName: {loinc_long_common_name}")
    logger.info(f"daysFromStudyToNegCovidTest: {days_from_study_to_neg_covid_test}")
    logger.info(f"studyYear: {study_year}")
    logger.info(f"studyYearShifted: {study_year_shifted}")
    logger.info(f"daysFromStudyToPosCovidTest: {days_from_study_to_pos_covid_test}")
    logger.info(f"ageAtImaging: {age_at_imaging}")
    logger.info(f"loincContrast: {loinc_contrast}")
    logger.info(f"projectId: {project_id}")
    logger.info(f"bodyPartExamined: {body_part_examined}")
    logger.info(f"loincCode: {loinc_code}")
    logger.info(f"sex: {sex}")
    logger.info(f"race: {race}")
    logger.info(f"ageAtIndex: {age_at_index}")
    logger.info(f"indexEvent: {index_event}")
    logger.info(f"covid19Positive: {covid19_positive}")
    logger.info(f"ethnicity: {ethnicity}")
    logger.info(f"dataFormat: {data_format}")
    logger.info(f"dataType: {data_type}")
    logger.info(f"dataCategory: {data_category}")
    logger.info(f"dataFileAnnotationName: {data_file_annotation_name}")
    logger.info(f"dataFileAnnotationMethod: {data_file_annotation_method}")
    logger.info(f"midrcMRALEScore: {midrc_mRALE_score}")
    logger.info(f"testDaysFromIndex: {test_days_from_index}")
    logger.info(f"testMethod: {test_method}")
    logger.info(f"testName: {test_name}")
    logger.info(f"testDaysFromIndex: {test_days_from_index}")
    logger.info(f"testResultText: {test_result_text}")
    logger.info(f"conditionName: {condition_name}")
    logger.info(f"conditionCodeSystem: {condition_code_system}")
    logger.info(f"conditionCode: {condition_code}")
    logger.info(f"daysToConditionStart: {days_to_condition_start}")
    logger.info(f"daysToConditionEnd: {days_to_condition_end}")
    logger.info(f"daysToMedicationStart: {days_to_medication_start}")
    logger.info(f"doseSequenceNumber: {dose_sequence_number}")
    logger.info(f"medicationCode: {medication_code}")
    logger.info(f"medicationManufacturer: {medication_manufacturer}")
    logger.info(f"medicationName: {medication_name}")
    logger.info(f"medicationType: {medication_type}")
    logger.info(f"annotationMethod: {annotation_method}")
    logger.info(f"annotationId: {annotation_id}")
    logger.info(f"breathingSupportType: {breathing_support_type}")
    logger.info(f"sourceNode: {source_node}")
    logger.info(f"first: {first}")
    logger.info(f"offset: {offset}")
    logger.info(f"outDir: {out_dir}")

    option_values = [
        md.cred,
        study_modality,
        study_description,
        loinc_method,
        midrc_type.value,
        loinc_system,
        loinc_long_common_name,
        days_from_study_to_neg_covid_test,
        study_year,
        study_year_shifted,
        days_from_study_to_pos_covid_test,
        age_at_imaging,
        loinc_contrast,
        project_id,
        body_part_examined,
        loinc_code,
        sex,
        race,
        age_at_index,
        index_event,
        covid19_positive,
        ethnicity,
        data_format,
        data_type,
        data_category,
        data_file_annotation_name,
        data_file_annotation_method,
        midrc_mRALE_score,
        test_days_from_index,
        test_method,
        test_name,
        test_days_from_index,
        test_result_text,
        condition_name,
        condition_code_system,
        condition_code,
        days_to_condition_start,
        days_to_condition_end,
        days_to_medication_start,
        dose_sequence_number,
        medication_code,
        medication_manufacturer,
        medication_name,
        medication_type,
        annotation_method,
        annotation_id,
        breathing_support_type,
        source_node,
        first,
        offset,
        out_dir,
    ]

    params = ut.get_params(option_values)
    model = ut.TestCustomValidation(params = params)
    print(model.parse_json())


    # if preview:
    #     ut.generate_preview(out_dir)
    #     logger.info(f"generating preview data in {out_dir}")
    # else:
    #     model = md.MIDRIC(**params)
    #     filter_obj = model.get_query(params)

    #     sort_fields = [{"submitter_id": "asc"}]

    #     data = model.query_data(
    #         midrc_type=midrc_type.value,
    #         fields=None,
    #         filter_object=filter_obj,
    #         sort_fields=sort_fields,
    #         first=first,
    #         offset=offset,
    #     )
    #     model.download_data(data)


if __name__ == "__main__":
    app()
