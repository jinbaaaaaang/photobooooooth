import cv2 as cv
import numpy as np
from datetime import datetime


# 필터 관련 전역 상태
FILTER_NAMES = ["Original", "Soft", "Warm", "Cool", "Glow", "Film", "Pastel"]
current_filter = "Original"
show_filter_menu = False
filter_button_rect = None  # (x1, y1, x2, y2)
filter_menu_items = []  # [(name, (x1, y1, x2, y2)), ...]

# 좌우 반전 토글 버튼 상태
mirror_button_rect = None
is_mirrored = True


def apply_filter(frame, name):
    if name == "Soft":
        # 살짝 흐릿하고 밝아지는 느낌
        blur = cv.GaussianBlur(frame, (9, 9), 0)
        soft = cv.addWeighted(blur, 0.8, frame, 0.2, 10)
        return soft
    if name == "Warm":
        # 부드러운 따뜻한 톤: 살짝 밝게 + 주황빛만 살짝 추가
        warm = frame.astype(np.float32)
        # 전체를 약간 밝게
        warm = warm * 1.05 + 5
        # BGR에서 R 채널(붉은기)만 조금 올리고 B를 아주 살짝 줄인다.
        b, g, r = cv.split(warm)
        r = r + 12
        b = b - 5
        warm = cv.merge((b, g, r))
        return np.clip(warm, 0, 255).astype(np.uint8)
    if name == "Cool":
        # 차가운 톤 (파랑/시안 쪽으로 살짝 이동)
        b, g, r = cv.split(frame)
        b = cv.add(b, 15)
        r = cv.subtract(r, 10)
        cool = cv.merge((b, g, r))
        return np.clip(cool, 0, 255).astype(np.uint8)
    if name == "Glow":
        # 하이라이트가 살짝 번지면서, 아주 약간 푸른 느낌이 나는 글로우
        blur = cv.GaussianBlur(frame, (21, 21), 0)
        glow = cv.addWeighted(frame, 0.7, blur, 0.5, 10)
        # B 채널만 살짝 올려서 푸른 기운을 아주 약하게 추가
        b, g, r = cv.split(glow.astype(np.float32))
        b = b + 8
        glow = cv.merge((b, g, r))
        return np.clip(glow, 0, 255).astype(np.uint8)
    if name == "Film":
        # 필름 느낌: 약간의 대비 상승 + 살짝 색감 보정 + 아주 약한 그레인
        film = frame.astype(np.float32)
        film = cv.addWeighted(film, 1.1, np.zeros_like(film), 0, -10)
        b, g, r = cv.split(film)
        r = r + 5
        b = b - 5
        film = cv.merge((b, g, r))
        noise = np.random.normal(0, 5, film.shape).astype(np.float32)
        film = film + noise
        return np.clip(film, 0, 255).astype(np.uint8)
    if name == "Pastel":
        # 파스텔 느낌: 채도와 대비를 줄이고 전체를 밝게
        pastel = frame.astype(np.float32)
        pastel = cv.addWeighted(pastel, 0.7, np.full_like(pastel, 255, dtype=np.float32), 0.3, 0)
        pastel = cv.GaussianBlur(pastel, (5, 5), 0)
        return np.clip(pastel, 0, 255).astype(np.uint8)


def draw_round_rect(img, pt1, pt2, radius, color):
    x1, y1 = pt1
    x2, y2 = pt2
    if radius <= 0:
        cv.rectangle(img, (x1, y1), (x2, y2), color, -1)
        return
    radius = int(radius)
    # 가운데 직사각형
    cv.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    # 네 모서리 원
    cv.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
    cv.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
    cv.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    cv.circle(img, (x2 - radius, y2 - radius), radius, color, -1)


def on_mouse(event, x, y, flags, param):
    global show_filter_menu, current_filter, is_mirrored
    # 버튼이 잘 눌리도록, 마우스 왼쪽 버튼이 눌렸을 때 바로 처리한다.
    if event != cv.EVENT_LBUTTONDOWN:
        return

    # 필터 메뉴가 열려 있으면 메뉴에서 항목 선택
    if show_filter_menu:
        for name, (x1, y1, x2, y2) in filter_menu_items:
            if x1 <= x <= x2 and y1 <= y <= y2:
                current_filter = name
                show_filter_menu = False
                return
        # 메뉴 밖을 클릭하면 메뉴 닫기
        show_filter_menu = False
        return

    # 메뉴가 닫혀 있을 때 필터/미러 버튼 클릭 여부 확인
    if filter_button_rect is not None:
        x1, y1, x2, y2 = filter_button_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            show_filter_menu = True
            return

    if mirror_button_rect is not None:
        x1, y1, x2, y2 = mirror_button_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            is_mirrored = not is_mirrored


def main():  # 프로그램의 메인 실행 함수이다.
    cap = cv.VideoCapture(0)  # 기본 카메라(보통 웹캠)를 연다.

    # 카메라 해상도/프레임으로 Writer 설정용
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))  # 카메라 영상의 가로 해상도를 가져온다.
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))  # 카메라 영상의 세로 해상도를 가져온다.
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0  # 카메라 FPS를 가져오고, 값이 없으면 30.0을 사용한다.

    is_recording = False  # 처음에는 녹화하지 않는 Preview 모드로 시작한다.
    writer = None  # 동영상 저장 객체는 아직 만들지 않았으므로 None으로 둔다.
    output_path = None  # 저장 파일 이름도 아직 없으므로 None으로 둔다.

    bar_height = 120  # 아래쪽 컨트롤 바 높이 (버튼 영역, 조금 더 크게)

    cv.namedWindow("Video Recorder")
    cv.setMouseCallback("Video Recorder", on_mouse)

    while True:  # 프로그램이 종료될 때까지 계속 반복한다.
        ret, frame = cap.read()  # 카메라에서 한 프레임을 읽어온다.
        if not ret:  # 프레임을 정상적으로 읽지 못했는지 확인한다.
            break  # 읽기 실패 시 반복을 종료한다.

        # 웹캠 원본은 좌우가 뒤집혀 보일 수 있으므로, 설정에 따라 좌우 반전한다.
        if is_mirrored:
            frame = cv.flip(frame, 1)

        # 필터 미리보기를 위해 원본 프레임을 따로 보관해 둔다.
        raw_frame = frame.copy()

        # 카메라 프레임에 현재 선택된 필터를 먼저 적용한다.
        filtered = apply_filter(frame, current_filter)

        # 방어적으로 결과를 확인하고, 문제 있으면 원본 프레임을 그대로 사용한다.
        if filtered is None or not isinstance(filtered, np.ndarray):
            frame = frame
        else:
            frame = filtered

        # 혹시 필터 처리 결과가 그레이스케일(2차원)로 바뀐 경우를 대비해 항상 3채널로 맞춘다.
        if frame.ndim == 2:
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

        # 전체 배경을 기본색(#CBDFEA)으로 채운 뒤, 그 위에 카메라 영상을 올린다.
        # OpenCV는 BGR 순서를 쓰므로, RGB(#CBDFEA)는 BGR(234, 223, 203)이다.
        bg = np.full(frame.shape, (234, 223, 203), dtype=np.uint8)
        bg[0 : height - bar_height, 0:width] = frame[0 : height - bar_height, 0:width]
        frame = bg

        # ===== 아래쪽 컨트롤 바 (심플한 Photo Booth 스타일) =====
        bar_y = height - bar_height
        # 하단 바 전체를 기본색(#CBDFEA)으로 채운다. (BGR: 234, 223, 203)
        cv.rectangle(
            frame,
            (0, bar_y),
            (width, height),
            (234, 223, 203),  # 기본색 #CBDFEA (BGR)
            -1,
        )
        # 바 위쪽 경계선 (기본색보다 조금 어두운 톤)
        cv.line(frame, (0, bar_y), (width, bar_y), (180, 200, 212), 1)

        # 왼쪽의 필터 버튼 ("Filter" 글자를 감싸는 캡슐 형태 버튼)
        btn_h = 40
        btn_w = 140
        btn_x1 = 50
        btn_y1 = bar_y + (bar_height - btn_h) // 2
        btn_x2 = btn_x1 + btn_w
        btn_y2 = btn_y1 + btn_h
        radius = btn_h // 2
        center_y = (btn_y1 + btn_y2) // 2

        # "Filter" 글자를 감싸는 캡슐 버튼
        left_center = (btn_x1 + radius, center_y)
        right_center = (btn_x2 - radius, center_y)
        # 안쪽과 테두리가 모두 같은 밝은 색 (하얀 캡슐 버튼 느낌)
        color = (245, 245, 245)
        thickness = 2

        # 먼저 내부를 바 배경색과 비슷한 밝은 색으로 채운다.
        cv.rectangle(
            frame,
            (left_center[0], btn_y1),
            (right_center[0], btn_y2),
            (245, 245, 245),
            -1,
        )
        cv.circle(frame, left_center, radius, (245, 245, 245), -1)
        cv.circle(frame, right_center, radius, (245, 245, 245), -1)

        # 그 위에 같은 색으로 윤곽선을 한 번만 그린다.
        # 양쪽 원 테두리
        cv.circle(frame, left_center, radius, color, thickness)
        cv.circle(frame, right_center, radius, color, thickness)
        # 위/아래를 잇는 직선
        top_y = center_y - radius
        bottom_y = center_y + radius
        cv.line(
            frame,
            (left_center[0], top_y),
            (right_center[0], top_y),
            color,
            thickness,
        )
        cv.line(
            frame,
            (left_center[0], bottom_y),
            (right_center[0], bottom_y),
            color,
            thickness,
        )

        # 가운데에 "Filter" 텍스트
        text = "Filter"
        (text_w, text_h), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = btn_x1 + (btn_w - text_w) // 2
        text_y = btn_y1 + (btn_h + text_h) // 2
        cv.putText(
            frame,
            text,
            (text_x, text_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (53, 57, 75),
            2,
            cv.LINE_AA,
        )

        # 마우스 콜백에서 사용할 필터 버튼 영역을 전역 변수에 저장
        global filter_button_rect, mirror_button_rect, filter_menu_items
        filter_button_rect = (btn_x1, btn_y1, btn_x2, btn_y2)

        # Filter 버튼 옆에 같은 모양의 Mirror 버튼을 배치
        mirror_btn_h = btn_h
        mirror_btn_w = btn_w
        gap = 20
        mirror_btn_x1 = btn_x2 + gap
        mirror_btn_y1 = btn_y1
        mirror_btn_x2 = mirror_btn_x1 + mirror_btn_w
        mirror_btn_y2 = btn_y2
        mirror_left_center = (mirror_btn_x1 + radius, center_y)
        mirror_right_center = (mirror_btn_x2 - radius, center_y)

        # Mirror 캡슐 내부 채우기 (Filter와 동일 색)
        cv.rectangle(
            frame,
            (mirror_left_center[0], mirror_btn_y1),
            (mirror_right_center[0], mirror_btn_y2),
            (245, 245, 245),
            -1,
        )
        cv.circle(frame, mirror_left_center, radius, (245, 245, 245), -1)
        cv.circle(frame, mirror_right_center, radius, (245, 245, 245), -1)

        # Mirror 캡슐 윤곽선
        cv.circle(frame, mirror_left_center, radius, color, thickness)
        cv.circle(frame, mirror_right_center, radius, color, thickness)
        cv.line(
            frame,
            (mirror_left_center[0], top_y),
            (mirror_right_center[0], top_y),
            color,
            thickness,
        )
        cv.line(
            frame,
            (mirror_left_center[0], bottom_y),
            (mirror_right_center[0], bottom_y),
            color,
            thickness,
        )

        mirror_text = "Mirror"
        (m_w, m_h), _ = cv.getTextSize(mirror_text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        m_x = mirror_btn_x1 + (mirror_btn_w - m_w) // 2
        m_y = mirror_btn_y1 + (mirror_btn_h + m_h) // 2
        cv.putText(
            frame,
            mirror_text,
            (m_x, m_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (53, 57, 75),
            2,
            cv.LINE_AA,
        )
        mirror_button_rect = (mirror_btn_x1, mirror_btn_y1, mirror_btn_x2, mirror_btn_y2)

        # 가운데 동그란 녹화 버튼 (다른 버튼들과 비슷한 톤, 바 높이 중앙 정렬)
        btn_center = (width // 2, bar_y + bar_height // 2)
        btn_radius = 34
        base_fill = (245, 245, 245)
        border_color = (200, 200, 200)
        icon_color = (53, 57, 75)

        # 공통 배경/테두리
        cv.circle(frame, btn_center, btn_radius, base_fill, -1)
        cv.circle(frame, btn_center, btn_radius, border_color, 2)

        if is_recording:
            cam_w, cam_h = 32, 20
        else:
            cam_w, cam_h = 28, 18

        cam_x1 = btn_center[0] - cam_w // 2
        cam_y1 = btn_center[1] - cam_h // 2
        cam_x2 = cam_x1 + cam_w
        cam_y2 = cam_y1 + cam_h

        # 카메라 본체는 흰색으로 채우고, 포인트색으로 테두리를 그려서 또렷하게 보이게 한다.
        cv.rectangle(frame, (cam_x1, cam_y1), (cam_x2, cam_y2), (255, 255, 255), -1)
        cv.rectangle(frame, (cam_x1, cam_y1), (cam_x2, cam_y2), icon_color, 2)

        # 렌즈: 녹화 중에는 꽉 찬 동그라미, Preview에는 테두리만 있는 동그라미
        lens_radius = 4 if is_recording else 3
        lens_thickness = -1 if is_recording else 2
        cv.circle(
            frame,
            (btn_center[0], btn_center[1]),
            lens_radius,
            icon_color,
            lens_thickness,
        )

        # 필터 메뉴가 열려 있으면, 하단 바 위에 필터 선택 창을 그린다.
        filter_menu_items = []
        if show_filter_menu:
            # 필터 미리보기가 들어가는 메뉴 (하단 UI와 색감을 통일, 둥근 모서리)
            menu_width = 260
            thumb_w, thumb_h = 80, 50
            # 카드 높이 = 썸네일 높이 + 위/아래 패딩 + 카드 간 마진
            vertical_padding = 10
            card_margin = 6
            menu_item_h = thumb_h + vertical_padding * 2 + card_margin
            menu_x1 = filter_button_rect[0]
            menu_x2 = menu_x1 + menu_width
            # 패널 전체 높이 계산 후, 위쪽이 화면 밖으로 나가지 않도록 보정한다.
            total_height = menu_item_h * len(FILTER_NAMES)
            ideal_menu_y2 = bar_y - 8
            ideal_menu_y1 = ideal_menu_y2 - total_height - 24
            menu_y1 = max(10, ideal_menu_y1)
            menu_y2 = menu_y1 + total_height + 24
            panel_bg = (240, 240, 240)  # 패널 배경 (버튼보다 살짝 어두운 회색)
            accent = (53, 57, 75)

            # 둥근 모서리 패널 배경
            draw_round_rect(frame, (menu_x1, menu_y1), (menu_x2, menu_y2), radius=16, color=panel_bg)
            # 각 필터 항목 (썸네일 + 이름)
            cur_y = menu_y1 + 12
            # 썸네일 생성용으로 상단 일부 영역만 잘라서 사용
            roi_h = min(thumb_h * 3, height - bar_height)
            base_thumb = cv.resize(
                raw_frame[0:roi_h, 0:roi_h],
                (thumb_w, thumb_h),
            )
            for name in FILTER_NAMES:
                # 카드 영역 (항목 사이에 card_margin 만큼 갭을 둔다)
                item_y1 = cur_y
                item_y2 = cur_y + menu_item_h - card_margin
                item_x1 = menu_x1 + 12
                item_x2 = menu_x2 - 12

                # 필터 카드 배경 (살짝 둥근 카드 형태)
                card_radius = 12
                # 기본 카드 배경은 패널보다 조금 더 밝게
                card_bg = (252, 252, 252)
                if name == current_filter:
                    card_bg = (234, 223, 203)  # 선택된 항목은 하단 바 색으로
                draw_round_rect(frame, (item_x1, item_y1), (item_x2, item_y2), card_radius, card_bg)

                # 썸네일: 현재 카메라 프레임에 필터를 적용한 작은 이미지
                # Original은 필터를 적용하지 않은 원본 썸네일을 그대로 사용한다.
                if name == "Original":
                    thumb = base_thumb.copy()
                else:
                    thumb = apply_filter(base_thumb.copy(), name)
                # 방어적으로 썸네일이 None 이거나 배열이 아니면 이 항목은 건너뛴다.
                if thumb is None or not isinstance(thumb, np.ndarray):
                    filter_menu_items.append(
                        (
                            name,
                            (item_x1, item_y1, item_x2, item_y2),
                        )
                    )
                    cur_y += menu_item_h
                    continue

                thumb_x1 = item_x1 + 10
                # 썸네일을 카드 안에서 세로로도 어느 정도 중앙에 오도록 배치
                thumb_y1 = item_y1 + vertical_padding
                thumb_x2 = thumb_x1 + thumb_w
                thumb_y2 = thumb_y1 + thumb_h
                frame[thumb_y1:thumb_y2, thumb_x1:thumb_x2] = thumb

                # 썸네일 오른쪽에 필터 이름
                # 텍스트를 카드 세로 중앙에 가깝게 배치
                text_center_y = (item_y1 + item_y2) // 2
                cv.putText(
                    frame,
                    name,
                    (thumb_x2 + 14, text_center_y + 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    accent,
                    1,
                    cv.LINE_AA,
                )

                filter_menu_items.append(
                    (
                        name,
                        (item_x1, item_y1, item_x2, item_y2),
                    )
                )
                # 다음 카드로 내려갈 때 약간의 마진을 포함해 이동
                cur_y += menu_item_h

        # 녹화 중이면, UI가 그려진 프레임을 파일에 저장한다.
        if is_recording and writer is not None:
            writer.write(frame)

        cv.imshow("Video Recorder", frame)  # 현재 프레임을 화면 창에 출력한다.

        key = cv.waitKey(1) & 0xFF  # 1ms 동안 키 입력을 기다리고 하위 8비트만 가져온다.
        if key == 27:  # ESC 키가 눌렸는지 확인한다.
            break  # ESC가 눌리면 반복문을 종료한다.
        elif key == ord(" "):  # Space 키가 눌렸는지 확인한다.
            is_recording = not is_recording  # Preview 모드와 Record 모드를 서로 전환한다.
            if is_recording:  # 전환 후 녹화 모드가 되었다면
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filter_suffix = current_filter.lower()
                output_path = f"rec_{timestamp}_{filter_suffix}.mp4"
                fourcc = cv.VideoWriter_fourcc(*"mp4v")
                writer = cv.VideoWriter(output_path, fourcc, fps, (width, height))
                print(f"녹화 시작: {output_path}")
            else:  # 전환 후 Preview 모드가 되었다면
                if writer is not None:  # 저장 객체가 존재하는지 확인한다.
                    writer.release()  # 동영상 파일 저장을 종료하고 자원을 해제한다.
                    writer = None  # writer 변수를 다시 None으로 초기화한다.
                    print(f"녹화 종료: {output_path}")  # 녹화 종료 메시지를 출력한다.

    cap.release()  # 카메라 장치를 해제한다.
    if writer is not None:  # 종료 시 writer가 아직 열려 있는지 확인한다.
        writer.release()  # 열려 있다면 저장 객체도 해제한다.
    cv.destroyAllWindows()  # OpenCV로 연 모든 창을 닫는다.


if __name__ == "__main__":  # 이 파일을 직접 실행했을 때만 아래 코드를 실행한다.
    main()  # main 함수를 호출해서 프로그램을 시작한다.