# -*- coding: utf-8 -*-
import csv

test_data = [
    ["image1.jpg", "Một ngày mới tràn đầy năng lượng", "Một ngay moi tran day nang luong", "Một ngày mới tràn đầy nâng lượng", "Môt ngày mới tràn đầy năng lượng"],
    ["image2.jpg", "Việt Nam đất nước tôi yêu", "Viet Nam dat nuoc toi yeu", "Việt Nam đất nước tôi yêu", "Việt Nam đât nước tôi yêu"],
    ["image3.jpg", "Cửa sổ nhìn ra biển xanh", "Cua so nhin ra bien xanh", "Cửa sổ nhìn ra biễn xanh", "Cửa sổ nhìn ra biển xanh"],
    ["image4.jpg", "Những cánh hoa đào rơi", "Nhung canh hoa dao roi", "Những cánh hoa đào rơi", "Những cành hoa đào rơi"],
    ["image5.jpg", "Tiếng chim hót líu lo", "Tieng chim hot liu lo", "Tiếng chim hót líu lo", "Tiêng chim hót líu lo"],
    ["image6.jpg", "Mưa rơi trên phố vắng", "Mua roi tren pho vang", "Mưa rơi trên phố vắng", "Mưa rơi trên phô vắng"],
    ["image7.jpg", "Bến xe đông người đợi", "Ben xe dong nguoi doi", "Bến xe đông người đơi", "Bến xe đông người đợi"],
    ["image8.jpg", "Sông Hồng nước chảy êm đềm", "Song Hong nuoc chay em dem", "Sông Hồng nước chảy êm đềm", "Sông Hông nước chảy êm đềm"],
    ["image9.jpg", "Phở thơm ngát góc phố", "Pho thom ngat goc pho", "Phở thơm ngát góc phố", "Phở thơm ngát gốc phố"],
    ["image10.jpg", "Hồ Gươm chiều thu tĩnh lặng", "Ho Guom chieu thu tinh lang", "Hồ Gươm chiều thu tĩnh lặng", "Hồ Gươm chiêu thu tĩnh lặng"]
]

# Generate 100 more variations by repeating with different filenames
extended_data = []
for i in range(1, 11):
    for row in test_data:
        new_row = row.copy()
        new_row[0] = f"batch{i}_{row[0]}"
        extended_data.append(new_row)

# Save to CSV with UTF-8 encoding
with open('test_ocr_data.csv', 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(extended_data)

print("Generated test_ocr_data.csv with 110 entries using UTF-8 encoding")

# Verify the encoding
with open('test_ocr_data.csv', 'r', encoding='utf-8-sig') as f:
    first_lines = [next(f) for _ in range(5)]
    print("\nFirst 5 lines preview:")
    print(''.join(first_lines))