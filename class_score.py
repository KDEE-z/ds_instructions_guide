EXCELENT_THRESHOLD = 80

class ClassScore:
    """クラスの成績を管理する"""
    
    def __init__(self, list_student_score:list[int]) -> None:
        """初期化処理"""
        self._list_student_score = list_student_score
    
    def num_excelent(self) -> None:
        """優の生徒の数を数える"""
        return sum(score >= EXCELENT_THRESHOLD for score in self._list_student_score)
    
class GradeScore:
    """学年の成績を管理するクラス"""
    
    def __init__(self, list_class_score: list[ClassScore]) -> None:
        """初期化処理"""
        self._list_class_score = list_class_score
        
    def num_excelent(self) -> int:
        """優の生徒の数を数える"""
        return sum(class_score.num_excelent() for class_score in self._list_class_score)
    
grade_score = GradeScore(
    [
        ClassScore([78, 90, 65, 80, 95]),
        ClassScore([85, 70, 75, 90, 80]),
        ClassScore([60, 75, 80, 85, 90])
    ]
)
num_excellent = grade_score.num_excelent()
print(num_excellent)