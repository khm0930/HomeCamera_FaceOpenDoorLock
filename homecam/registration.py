from flask import render_template, request, Blueprint

registration_bp = Blueprint('registration', __name__)

@registration_bp.route('/register', methods=['GET']) 
def render_register_page():
    return render_template('register.html')

@registration_bp.route('/register', methods=['POST'])
def process_user_registration():
    if request.method == 'POST':
        # 사용자 등록 로직을 처리0
        username = request.form['username']
        password = request.form['password']
        # Firebase 또는 다른 백엔드 서비스를 사용하여 사용자 등록 및 데이터베이스 업데이트 수행
        # 이 곳에 해당 코드를 추가하세요.

        # 사용자 등록에 성공했다면 리디렉션 또는 응답을 반환
        return '회원가입 완료!'